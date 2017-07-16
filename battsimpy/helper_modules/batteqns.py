# -*- coding:utf-8 -*-
"""Package of methods for battery model calculations.
"""
import numpy
from copy import deepcopy
from scipy.signal import filtfilt, butter
import scipy.interpolate


# New functions for full FVM based P2D model
def compute_deriv(func, x0):
    """
    Brute force method to calculate the numerical Jacobian from the residual
    function <func>.
    This is a super slow method, but effective for debugging your analytical
    Jacobian operator.
    """
    y0 = func(x0)

    J = numpy.zeros((len(x0), len(x0)), dtype='d')

    x_higher = deepcopy(x0)

    eps = 1e-8

    for ivar in range(len(x0)):

        x_higher[ivar] = x_higher[ivar] + eps

        # evaluate the function
        y_higher = func(x_higher)

        dy_dx = (y_higher - y0) / eps

        J[:, ivar] = dy_dx

        x_higher[ivar] = x0[ivar]

    return J


def right_side_coeffs(h_n, h_n1):
    """
    General 1D non-uniform mesh values for retrieving the boundary values.

    E.g., for the core and surface values of the particle concentration PDE.
    """
    a_n = h_n / (h_n1 * (h_n1 + h_n))
    b_n = -(h_n1 + h_n) / (h_n1 * h_n)
    c_n = (2.0*h_n + h_n1) / (h_n * (h_n1 + h_n))

    return a_n, b_n, c_n


def left_side_coeffs(h_n, h_n1):
    """
    General 1D non-uniform mesh values for retrieving the boundary values.

    E.g., for the core and surface values of the particle concentration PDE.
    """
    a_n = -(2.0*h_n + h_n1) / (h_n * (h_n1 + h_n))
    b_n = (h_n1 + h_n) / (h_n1 * h_n)
    c_n = -h_n / (h_n1 * (h_n1 + h_n))

    return a_n, b_n, c_n


def build_interp_2d(path, scalar=1.0):
    """
    Create the interpolator function for a 2-d data set.

    For example, this is used for table look-up for the electrolyte
    conductivity, which is a function of e-lyte concentration and temperature.

    e-lyte diffusivity and activity are also generally used here as well, etc.
    """
    raw_map = numpy.loadtxt(path, delimiter=",")

    v1 = raw_map[1:, 0]
    v2 = raw_map[0, 1:]

    dat_map = raw_map[1:, 1:] * scalar

    if v1[1] < v1[0]:
        v1 = numpy.flipud(v1)
        dat_map = numpy.flipud(dat_map)

    if v2[1] < v2[0]:
        v2 = numpy.flipud(v2)
        dat_map = numpy.fliplr(dat_map)

    v1_lims = [min(v1), max(v1)]

    return scipy.interpolate.RectBivariateSpline(v1, v2, dat_map), v1_lims


def ButterworthFilter(x, y, ff=0.2):
    """
    First order butterworth filter for smoothing an array.
    """
    b, a = butter(1, ff)
    fl = filtfilt(b, a, y)
    return fl


def get_smooth_Uref_data(Ua_path, Uc_path, ffa=0.4, ffc=0.2, filter_on=1):
    """
    Smooth the Uref data to aid in improving numerical stability.
    This should be verified by the user to ensure it is not changing the
    original Uref data beyond a tolerable amount (defined by the user).
    A linear interpolator class is output for Uref and dUref_dx for both anode
    and cathode.
    """
    # Load the data files
    uref_a_map = numpy.loadtxt(Ua_path, delimiter=',')
    uref_c_map = numpy.loadtxt(Uc_path, delimiter=',')

    if uref_a_map[1, 0] < uref_a_map[0, 0]:
        uref_a_map = numpy.flipud(uref_a_map)
    if uref_c_map[1, 0] < uref_c_map[0, 0]:
        uref_c_map = numpy.flipud(uref_c_map)

    xa = uref_a_map[:, 0]
    xc = uref_c_map[:, 0]

    # Smooth the signals
    if filter_on:
        xa_ref = numpy.linspace(min(xa), max(xa), 1000)
        xc_ref = numpy.linspace(min(xc), max(xc), 1000)

        Ua_butter = ButterworthFilter(xa_ref,
                                      numpy.interp(xa_ref, xa,
                                                   uref_a_map[:, 1]),
                                      ff=ffa)
        Uc_butter = ButterworthFilter(xc_ref,
                                      numpy.interp(xc_ref, xc,
                                                   uref_c_map[:, 1]),
                                      ff=ffc)
    else:
        xa_ref = xa
        xc_ref = xc

        Ua_butter = uref_a_map[:, 1]
        Uc_butter = uref_c_map[:, 1]

    # Create the interpolators
    Ua_intp = scipy.interpolate.interp1d(xa_ref, Ua_butter,
                                         kind='linear',
                                         fill_value=Ua_butter[-1],
                                         bounds_error=False)
    Uc_intp = scipy.interpolate.interp1d(xc_ref, Uc_butter,
                                         kind='linear',
                                         fill_value=Uc_butter[-1],
                                         bounds_error=False)

    duref_a = numpy.gradient(Ua_butter) / numpy.gradient(xa_ref)
    duref_c = numpy.gradient(Uc_butter) / numpy.gradient(xc_ref)

    dUa_intp = scipy.interpolate.interp1d(xa_ref, duref_a,
                                          kind='linear',
                                          fill_value=Ua_butter[-1],
                                          bounds_error=False)
    dUc_intp = scipy.interpolate.interp1d(xc_ref, duref_c,
                                          kind='linear',
                                          fill_value=Ua_butter[-1],
                                          bounds_error=False)

    return Ua_intp, Uc_intp, dUa_intp, dUc_intp


def nonlinspace(Rf, k, N):
    """
    Weighted grid spacing for spherical particle for more points at particle
    surface, than at the core.
    """
    r = numpy.zeros(N)
    for i in range(N):
        r[i] = (1./k)**(-i)

    if k != 1:
        r = max(r) - r
        r = r / max(r) * Rf
    else:
        r = r * Rf

    return r


def mid_to_edge(var_mid, x_e):
    """
    Interpolate a cell mid-point based array to the edge-points of the mesh.
    """
    var_edge = numpy.array([var_mid[0]]
                           + [var_mid[i]*var_mid[i+1]/(((x_e[i+1] - x_e[i])/((x_e[i+2] - x_e[i+1]) + (x_e[i+1] - x_e[i])))*var_mid[i+1] + (1 - ((x_e[i+1]-x_e[i])/((x_e[i+2]-x_e[i+1]) + (x_e[i+1] - x_e[i]))))*var_mid[i]) for i in range(len(var_mid) - 1)]
                           + [var_mid[-1]])

    return var_edge


def flux_mat_builder(N, x_m, vols, P):
    """
    Generate the basic matrix for the FVM flux matrix operator.
    """
    A = numpy.zeros([N, N], dtype='d')

    for i in range(1, N-1):
        A[i, i-1] = (1./vols[i]) * (P[i]) / (x_m[i] - x_m[i-1])
        A[i, i] = -(1./vols[i]) * (P[i]) / (x_m[i] - x_m[i-1]) \
            - (1./vols[i]) * (P[i+1]) / (x_m[i+1] - x_m[i])
        A[i, i+1] = (1./vols[i]) * (P[i+1]) / (x_m[i+1] - x_m[i])

    i = 0
    A[0, 0] = -(1./vols[i]) * (P[i+1]) / (x_m[i+1] - x_m[i])
    A[0, 1] = (1./vols[i]) * (P[i+1]) / (x_m[i+1] - x_m[i])

    i = N-1
    A[i, i-1] = (1./vols[i]) * (P[i]) / (x_m[i] - x_m[i-1])
    A[i, i] = -(1./vols[i]) * (P[i]) / (x_m[i] - x_m[i-1])

    return A


def grad_mat(N, x):
    """
    Generate a matrix that performs the centered difference gradient operator.
    """
    G = numpy.zeros([N, N])
    for i in range(1, N-1):
        G[i, [i-1, i+1]] = [-1./(x[i+1]-x[i-1]), 1./(x[i+1]-x[i-1])]

    # Edge boundary conditions
    G[0, [0, 1]] = [-1./(x[1]-x[0]), 1./(x[1]-x[0])]
    G[-1, [-2, -1]] = [-1./(x[-1]-x[-2]), 1./(x[-1]-x[-2])]

    return G


def get_ecm_params(ecm_params, ecmOrder, SOC, T):
    """
    Return a list of resistances for the ECM RC branches
    """
    # OCV lookup
    if ecm_params['ocv']['dim'] == '1D':
        OCV = ecm_params['ocv']['intp_func'](SOC)
    elif ecm_params['ocv']['dim'] == '2D':
        OCV = ecm_params['ocv']['intp_func'](SOC, T*numpy.ones_like(SOC))

    # Ohmic Resistance lookup
    if ecm_params['res_ohm']['dim'] == '1D':
        Rohm = ecm_params['res_ohm']['intp_func'](SOC)
    elif ecm_params['res_ohm']['dim'] == '2D':
        Rohm = ecm_params['res_ohm']['intp_func'](SOC, T*numpy.ones_like(SOC))

    # Resistance lookup
    Res = numpy.zeros(ecmOrder)
    for i_rc in range(ecmOrder):
        if ecm_params['res'][i_rc]['dim'] == '1D':
            Res[i_rc] = ecm_params['res'][i_rc]['intp_func'](SOC)
        elif ecm_params['res'][i_rc]['dim'] == '2D':
            Res[i_rc] = ecm_params['res'][i_rc]['intp_func'](SOC, T)

    # Time constant lookup
    Tau = numpy.zeros(ecmOrder)
    Cap = numpy.zeros(ecmOrder)
    for i_rc in range(ecmOrder):
        if ecm_params['tau'][i_rc]['dim'] == '1D':
            Tau[i_rc] = ecm_params['tau'][i_rc]['intp_func'](SOC)
        elif ecm_params['tau'][i_rc]['dim'] == '2D':
            Tau[i_rc] = ecm_params['tau'][i_rc]['intp_func'](
                SOC, T*numpy.ones_like(SOC))
        Cap[i_rc] = Tau[i_rc]/Res[i_rc]

    return OCV, Rohm, Res, Tau, Cap
