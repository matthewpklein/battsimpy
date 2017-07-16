import numpy
import numpy.linalg
import scipy.linalg
import scipy.interpolate

from scipy.signal import wiener, filtfilt, butter, gaussian
from scipy.ndimage import filters

from matplotlib import pyplot as plt
plt.style.use('classic')

from assimulo.solvers import IDA
from assimulo.problem import Implicit_Problem

from scipy.sparse.linalg import spsolve as sparseSolve
from scipy.sparse import csr_matrix as sparseMat
import scipy.sparse as sps
import scipy.sparse as sparse
import math
from copy import deepcopy

import params
import confreader

def compute_deriv( func, x0 ) :

    y0 = func(x0)

    J = numpy.zeros( (len(x0),len(x0)), dtype='d' )

    x_higher = deepcopy(x0)
    
    eps = 1e-8

    for ivar in range(len(x0)) :

        x_higher[ivar] = x_higher[ivar] + eps

        # evaluate the function
        y_higher = func(x_higher)

        dy_dx = (y_higher-y0) / eps

        J[:,ivar] = dy_dx

        x_higher[ivar] = x0[ivar]

    return J

def right_side_coeffs( h_n, h_n1 ) :

    a_n =             h_n    / ( h_n1 * (h_n1+h_n) )
    b_n = -(   h_n1 + h_n)   / ( h_n1 *  h_n       )
    c_n =  ( 2*h_n  + h_n1 ) / ( h_n  * (h_n1+h_n) )

    return a_n, b_n, c_n

def left_side_coeffs( h_n, h_n1 ) :

    a_n = -( 2*h_n  + h_n1 ) / ( h_n  * (h_n1+h_n) )
    b_n =  (   h_n1 + h_n)   / ( h_n1 *  h_n       )
    c_n = -           h_n    / ( h_n1 * (h_n1+h_n) )

    return a_n, b_n, c_n


def build_interp_2d( path, scalar=1.0 ) :

    raw_map = numpy.loadtxt( path, delimiter="," )

    v1 = raw_map[1:,0]
    v2 = raw_map[0,1:]

    dat_map = raw_map[1:,1:]*scalar

    if v1[1] < v1[0] :
        v1 = numpy.flipud( v1 )
        dat_map = numpy.flipud(dat_map)

    if v2[1] < v2[0] :
        v2 = numpy.flipud( v2 )
        dat_map = numpy.fliplr(dat_map)

    return scipy.interpolate.RectBivariateSpline( v1, v2, dat_map )    


def ButterworthFilter( x, y, ff=0.2 ) :
    b, a = butter(1, ff)
    fl = filtfilt( b, a, y )
    return fl

def get_smooth_Uref_data( Ua_path, Uc_path, ffa=0.4, ffc=0.2, filter_on=1 ) :
    """
    Smooth the Uref data to aid in improving numerical stability.
    This should be verified by the user to ensure it is not changing the original
    Uref data beyond a tolerable amount (defined by the user).
    A linear interpolator class is output for Uref and dUref_dx for both anode 
    and cathode.
    """
    ## Load the data files
    uref_a_map = numpy.loadtxt( Ua_path, delimiter=',' )
    uref_c_map = numpy.loadtxt( Uc_path, delimiter=',' )

    if uref_a_map[1,0] < uref_a_map[0,0] :
        uref_a_map = numpy.flipud( uref_a_map )
    if uref_c_map[1,0] < uref_c_map[0,0] :
        uref_c_map = numpy.flipud( uref_c_map )

    xa = uref_a_map[:,0]
    xc = uref_c_map[:,0]

#    big_xa = numpy.linspace( xa[0], xa[-1], 300 )
#    big_xc = numpy.linspace( xc[0], xc[-1], 300 )

#    big_Ua = numpy.interp( big_xa, xa, uref_a_map[:,1] )
#    big_Uc = numpy.interp( big_xc, xc, uref_c_map[:,1] )

#    numpy.savetxt( bsp_dir + '/data/Model_v1/Model_Pars/solid/thermodynamics/uref_anode_bigx.csv', numpy.array([big_xa, big_Ua]).T, delimiter=',' )
#    numpy.savetxt( bsp_dir + '/data/Model_v1/Model_Pars/solid/thermodynamics/uref_cathode_bigx.csv', numpy.array([big_xc, big_Uc]).T, delimiter=',' )

    ## Smooth the signals
    if filter_on :
        Ua_butter = ButterworthFilter( xa, uref_a_map[:,1], ff=ffa )
        Uc_butter = ButterworthFilter( xc, uref_c_map[:,1], ff=ffc )
    else :
        Ua_butter = uref_a_map[:,1]
        Uc_butter = uref_c_map[:,1]

    ## Create the interpolators
    Ua_intp = scipy.interpolate.interp1d( xa, Ua_butter, kind='linear', fill_value=Ua_butter[-1], bounds_error=False )
    Uc_intp = scipy.interpolate.interp1d( xc, Uc_butter, kind='linear', fill_value=Uc_butter[-1], bounds_error=False )

#    duref_a_map = numpy.gradient( uref_a_map[:,1] ) / numpy.gradient( xa )
#    duref_c_map = numpy.gradient( uref_c_map[:,1] ) / numpy.gradient( xc )

    duref_a = numpy.gradient( Ua_butter ) / numpy.gradient( xa )
    duref_c = numpy.gradient( Uc_butter ) / numpy.gradient( xc )

    dUa_intp = scipy.interpolate.interp1d( xa, duref_a, kind='linear' )
    dUc_intp = scipy.interpolate.interp1d( xc, duref_c, kind='linear' )

#    # Plot the Uref data for verification
#    plt.figure()
#    plt.plot( xa, uref_a_map[:,1], label='Ua map' )
#    plt.plot( xc, uref_c_map[:,1], label='Uc map' )

##        plt.plot( xa, Ua_butter, label='Ua butter' )
##        plt.plot( xc, Uc_butter, label='Uc butter' )

#    plt.plot( xa, self.uref_a(xa), label='Ua interp lin' )
#    plt.plot( xc, self.uref_c(xc), label='Uc interp lin' )
#    plt.legend()

#    plt.figure()
#    plt.plot( xa, duref_a_map, label='dUa map' )
#    plt.plot( xc, duref_c_map, label='dUc map' )

##        plt.plot( xa, duref_a_b, label='dUa B' )
##        plt.plot( xc, duref_c_b, label='dUc B' )

#    plt.plot( xa, self.duref_a_interp(xa), label='dUa interp butter' )
#    plt.plot( xc, self.duref_c_interp(xc), label='dUc interp butter' )
#    plt.legend()

#    plt.show()

    return Ua_intp, Uc_intp, dUa_intp, dUc_intp


def nonlinspace( Rf,k,N ) :

    r = numpy.zeros(N)
    for i in range(N) :
        r[i] = (1./k)**(-i)

    if k!=1 :
        r=max(r)-r
        r=r/max(r)*Rf
    else :
        r=r*Rf

    return r

def mid_to_edge( var_mid, x_e ) :

    var_edge = numpy.array( [var_mid[0]] + [ var_mid[i]*var_mid[i+1]/( ((x_e[i+1]-x_e[i])/((x_e[i+2]-x_e[i+1])+(x_e[i+1]-x_e[i])))*var_mid[i+1] + (1- ((x_e[i+1]-x_e[i])/((x_e[i+2]-x_e[i+1])+(x_e[i+1]-x_e[i]))))*var_mid[i] ) for i in range(len(var_mid)-1) ] + [var_mid[-1]] )

    return var_edge

def flux_mat_builder( N, x_m, vols, P ) :

    A = numpy.zeros([N,N], dtype='d')

    for i in range(1,N-1) :

        A[i,i-1] =  (1./vols[i]) * (P[i  ]) / (x_m[i  ] - x_m[i-1])
        A[i,i  ] = -(1./vols[i]) * (P[i  ]) / (x_m[i  ] - x_m[i-1]) - (1./vols[i]) * (P[i+1]) / (x_m[i+1] - x_m[i])
        A[i,i+1] =  (1./vols[i]) * (P[i+1]) / (x_m[i+1] - x_m[i  ])

    i=0
    A[0,0] = -(1./vols[i]) * (P[i+1]) / (x_m[i+1] - x_m[i])
    A[0,1] =  (1./vols[i]) * (P[i+1]) / (x_m[i+1] - x_m[i])

    i=N-1
    A[i,i-1] =  (1./vols[i]) * (P[i]) / (x_m[i] - x_m[i-1])
    A[i,i  ] = -(1./vols[i]) * (P[i]) / (x_m[i] - x_m[i-1])

    return A

def grad_mat( N, x ) :

    G = numpy.zeros( [N,N] )
    for i in range(1,N-1) :
        G[i,[i-1, i+1]] = [ -1./(x[i+1]-x[i-1]), 1./(x[i+1]-x[i-1]) ]
    G[0,[0,1]] = [-1./(x[1]-x[0]),1./(x[1]-x[0])]
    G[-1,[-2,-1]] = [-1./(x[-1]-x[-2]),1./(x[-1]-x[-2])]

    return G
#
#class params() :
#    def __init__(self,Na, Ns, Nc, Nra, Nrc, La,Ls,Lc, Ra, Rc, Ac, bsp_dir) :
#
#        self.bsp_dir = bsp_dir
#
#        self.Ac = Ac # Cell coated area, [m^2]
#
#        ### Control volumes and node points (mid node points and edge node points)
#        self.Na, self.Ns, self.Nc = Na, Ns, Nc
#        self.La, self.Ls, self.Lc = La, Ls, Lc
#
#        self.N = Na + Ns + Nc
#        self.X = La+Ls+Lc
#        N = self.N
#
#        self.x_e  = numpy.array(list(numpy.linspace( 0.0, La, Na+1 )) + list(numpy.linspace( La+Ls/float(Ns), Ls+La-Ls/float(Ns), Ns-1 )) + list(numpy.linspace( La+Ls, La+Ls+Lc, Nc+1 )))
#        self.x_m  = numpy.array( [ 0.5*(self.x_e[i+1]+self.x_e[i]) for i in range(N) ], dtype='d'  )
#        self.vols = numpy.array( [ (self.x_e[i+1] - self.x_e[i]) for i in range(N)], dtype='d' )
#
#        # Radial mesh
#        self.Nra = Nra
#        self.Nrc = Nrc
#
#        k=0.9999#0.85
#        self.r_e_a  = nonlinspace( Ra, k, Nra+1 )
#        self.r_m_a  = numpy.array( [ 0.5*(self.r_e_a[i+1]+self.r_e_a[i]) for i in range(Nra) ], dtype='d'  )
#        self.r_e_c  = nonlinspace( Rc, k, Nrc+1 )
#        self.r_m_c  = numpy.array( [ 0.5*(self.r_e_c[i+1]+self.r_e_c[i]) for i in range(Nrc) ], dtype='d'  )
#        self.vols_ra_m = numpy.array( [ 1/3.*(self.r_e_a[i+1]**3 - self.r_e_a[i]**3) for i in range(Nra)], dtype='d' )
#        self.vols_rc_m = numpy.array( [ 1/3.*(self.r_e_c[i+1]**3 - self.r_e_c[i]**3) for i in range(Nrc)], dtype='d' )
#
#        # Useful sub-meshes for the phi_s functions
#        self.x_m_a = self.x_m[:Na]
#        self.x_m_c = self.x_m[-Nc:]
#        self.x_e_a = self.x_e[:Na+1]
#        self.x_e_c = self.x_e[-Nc-1:]
#
#        self.vols_a = self.vols[:Na]
#        self.vols_c = self.vols[-Nc:]
#
#        self.num_diff_vars = self.N + self.Nra*self.Na + self.Nrc*self.Nc+1
#        self.num_algr_vars = self.Na+self.Nc + self.N + self.Na+self.Nc
#
#        ### Volume fraction vectors and matrices for effective parameters
##        self.La, self.Ls, self.Lc = self.Na*X/self.N, self.Ns*X/self.N, self.Nc*X/self.N
##        self.Na, self.Ns, self.Nc = Na, Ns, Nc
#        eps_a = 0.27
#        eps_s = 0.43
#        eps_c = 0.21
#        ba, bs, bc = 1.3, 0.5, 0.5 #0.95, 0.5, 0.35
#
#        eps_a_vec = [ eps_a for i in range(Na) ] # list( eps_a + eps_a/2.*numpy.sin(numpy.linspace(0.,Na/4,Na)) ) # list(eps_a + eps_a*numpy.random.randn(Na)/5.) #
#        eps_s_vec = [ eps_s for i in range(Ns) ]
#        eps_c_vec = [ eps_c for i in range(Nc) ] # list( eps_c + eps_c/2.*numpy.sin(numpy.linspace(0.,Nc/4,Nc)) ) # list(eps_c + eps_c*numpy.random.randn(Nc)/5.) #
#
#        self.eps_a_vec, self.eps_s_vec, self.eps_c_vec = eps_a_vec, eps_s_vec, eps_c_vec
#
#        self.eps_m   = numpy.array( eps_a_vec + eps_s_vec + eps_c_vec, dtype='d' )
#        self.k_m     = 1./self.eps_m
#        self.eps_mb  = numpy.array( [ ea**ba for ea in eps_a_vec ] + [ es**bs for es in eps_s_vec ] + [ ec**bc for ec in eps_c_vec ], dtype='d' )
#        self.eps_eff = numpy.array( [ ea**(1.+ba) for ea in eps_a_vec ] + [ es**(1.+bs) for es in eps_s_vec ] + [ ec**(1.+bc) for ec in eps_c_vec ], dtype='d' )
#
#        self.eps_a_eff = self.eps_eff[:Na]
#        self.eps_c_eff = self.eps_eff[-Nc:]
#
#        self.K_m = numpy.diag( self.k_m )
#
#        t_plus = 0.455
#        F = 96485.0
#
#        self.t_plus = t_plus
#        self.F = F
#        self.R_gas = 8.314
#
#        self.Rp_a = Ra
#        self.Rp_c = Rc
#
#        eps_f_a = eps_a - eps_a*1.
#        eps_f_c = eps_c - eps_c*1.
#
#        as_a = 3.*(1.0-numpy.array(eps_a_vec, dtype='d')-eps_f_a)/self.Rp_a
#        as_c = 3.*(1.0-numpy.array(eps_c_vec, dtype='d')-eps_f_c)/self.Rp_c
#        self.as_a = as_a
#        self.as_c = as_c
#
#        EASA_a = as_a[0] #1.70*(2.25*100**3) #*numpy.ones_like( eps_a_vec )
#        EASA_c = as_c[0] #0.26*(4.74*100**3) #*numpy.ones_like( eps_c_vec )
#
#        self.as_a_mean = 1./self.La*sum( [ asa*v for asa,v in zip(as_a, self.vols[:Na]) ] )
#        self.as_c_mean = 1./self.Lc*sum( [ asc*v for asc,v in zip(as_c, self.vols[-Nc:]) ] )
#
##        print 'asa diff', self.as_a_mean - as_a[0]
##        print 'asc diff', self.as_c_mean - as_c[0]
#
#        self.sig_a = 1000.0 # [S/m]
#        self.sig_c = 139.0 # [S/m]
#
#        self.sig_a_eff = self.sig_a * (1.-self.eps_a_eff)
#        self.sig_c_eff = self.sig_c * (1.-self.eps_c_eff)
#
#        self.ce_nom = 1000.0
#
#        # Interpolators for De, ke
#        self.De_intp  = build_interp_2d( bsp_dir+'data/Model_v1/Model_Pars/electrolyte/De_const.csv', scalar=1. )
#        self.ke_intp  = build_interp_2d( bsp_dir+'data/Model_v1/Model_Pars/electrolyte/kappa_const.csv' )
#
#        # Load the Ds data files
#        Dsa_map = numpy.loadtxt( bsp_dir+'data/Model_v1/Model_Pars/solid/diffusion/Ds_anode_const.csv', delimiter="," )
#        Dsc_map = numpy.loadtxt( bsp_dir+'data/Model_v1/Model_Pars/solid/diffusion/Ds_cathode_const.csv', delimiter="," )
#
#        if Dsa_map[1,0] < Dsa_map[0,0] :
#            Dsa_map = numpy.flipud( Dsa_map )
#        if Dsc_map[1,0] < Dsc_map[0,0] :
#            Dsc_map = numpy.flipud( Dsc_map )
#
#        ## Create the interpolators
#        self.Dsa_intp = scipy.interpolate.interp1d( Dsa_map[:,0], Dsa_map[:,1], kind='linear', fill_value=Dsa_map[-1,1], bounds_error=False  )
#        self.Dsc_intp = scipy.interpolate.interp1d( Dsc_map[:,0], Dsc_map[:,1], kind='linear', fill_value=Dsc_map[-1,1], bounds_error=False )
#
#        Dsa = numpy.mean(Dsa_map[:,1])
#        Dsc = numpy.mean(Dsc_map[:,1])
#        self.Dsa = Dsa
#        self.Dsc = Dsc
#
#        self.csa_max = 2.25*100**3/(96485.0/3.6/(365.0)) #30555.0 # [mol/m^3]
#        self.csc_max = 4.74*100**3/(96485.0/3.6/(282.0)) #51554.0 # [mol/m^3]
#
#        ### OCV
##        Ua_path = bsp_dir+'data/Model_v1/Model_Pars/solid/thermodynamics/uref_anode_bds.csv'
##        Uc_path = bsp_dir+'data/Model_v1/Model_Pars/solid/thermodynamics/uref_cathode_bds.csv'
#        Ua_path = bsp_dir+'data/Model_v1/Model_Pars/solid/thermodynamics/Un_simba_20170512.csv'
#        Uc_path = bsp_dir+'data/Model_v1/Model_Pars/solid/thermodynamics/Up_simba_20170512.csv'
##        Ua_path = bsp_dir+'data/Model_v1/Model_Pars/solid/thermodynamics/uref_anode_bigx.csv'
##        Uc_path = bsp_dir+'data/Model_v1/Model_Pars/solid/thermodynamics/uref_cathode_bigx.csv'
#
#        self.uref_a, self.uref_c, self.duref_a, self.duref_c = get_smooth_Uref_data( Ua_path, Uc_path, filter_on=0 ) #ffa=0.4, ffc=0.2 )
#
#        ### Reaction kinetics parameters
#        # Interpolators for io_a, io_c
##        print 'scalar = EASA_a/self.as_a[0]:', EASA_a/self.as_a[0]
#
#        self.ioa_interp = build_interp_2d( bsp_dir+'data/Model_v1/Model_Pars/solid/kinetics/io_anode_const.csv', scalar=0.999 ) #, scalar=EASA_a/self.as_a[0] )
#        self.ioc_interp = build_interp_2d( bsp_dir+'data/Model_v1/Model_Pars/solid/kinetics/io_cathode_const.csv', scalar=0.999 ) #, scalar=EASA_c/self.as_c[0] )
#
##        self.io_a = 10.0 # [A/m^2]
##        self.io_c = 10.0 # [A/m^2]
#
#        ### params for thermal calcs 
#        self.h, self.Aconv, self.rho, self.Cp = 100.0, 30., 2250.*self.X, 1200. # conv heat coeff [W/m^2-K], conv area ratio (Aconv/Acoat) [m^2/m^2], density per coated area [kg/m^2], specific heat capacity [J/kg-K]


class MyProblem( Implicit_Problem ) :

    def __init__(self, p, y0, yd0 ) :

        self.p = p

        self.Ac = p.Ac

        self.T_amb = 30.+273.15
        self.T     = 30.+273.15  # Cell temperature, [K]

        self.phie_mats()
        self.phis_mats()
        self.cs_mats()

        ### System indices
        self.ce_inds   = range( p.N )
        self.ce_inds_r = numpy.reshape( self.ce_inds, [len(self.ce_inds),1] )
        self.ce_inds_c = numpy.reshape( self.ce_inds, [1,len(self.ce_inds)] )

        self.csa_inds = range( p.N, p.N + (p.Na*p.Nra) )
        self.csa_inds_r = numpy.reshape( self.csa_inds, [len(self.csa_inds),1] )
        self.csa_inds_c = numpy.reshape( self.csa_inds, [1,len(self.csa_inds)] )

        self.csc_inds = range( p.N + (p.Na*p.Nra), p.N + (p.Na*p.Nra) + (p.Nc*p.Nrc) )
        self.csc_inds_r = numpy.reshape( self.csc_inds, [len(self.csc_inds),1] )
        self.csc_inds_c = numpy.reshape( self.csc_inds, [1,len(self.csc_inds)] )

        self.T_ind = p.N + (p.Na*p.Nra) + (p.Nc*p.Nrc)

        c_end = p.N + (p.Na*p.Nra) + (p.Nc*p.Nrc) + 1

        self.ja_inds = range(c_end, c_end+p.Na)
        self.ja_inds_r = numpy.reshape( self.ja_inds, [len(self.ja_inds),1] )
        self.ja_inds_c = numpy.reshape( self.ja_inds, [1,len(self.ja_inds)] )

        self.jc_inds = range(c_end+p.Na, c_end+p.Na +p.Nc)
        self.jc_inds_r = numpy.reshape( self.jc_inds, [len(self.jc_inds),1] )
        self.jc_inds_c = numpy.reshape( self.jc_inds, [1,len(self.jc_inds)] )
        
        self.pe_inds   = range( c_end+p.Na+p.Nc, c_end+p.Na+p.Nc +p.N )
        self.pe_inds_r = numpy.reshape( self.pe_inds, [len(self.pe_inds),1] )
        self.pe_inds_c = numpy.reshape( self.pe_inds, [1,len(self.pe_inds)] )

        self.pe_a_inds = range( c_end+p.Na+p.Nc, c_end+p.Na+p.Nc +p.Na )
        self.pe_a_inds_r = numpy.reshape( self.pe_a_inds, [len(self.pe_a_inds),1] )
        self.pe_a_inds_c = numpy.reshape( self.pe_a_inds, [1,len(self.pe_a_inds)] )

        self.pe_c_inds = range( c_end+p.Na+p.Nc +p.Na+p.Ns, c_end+p.Na+p.Nc +p.N )
        self.pe_c_inds_r = numpy.reshape( self.pe_c_inds, [len(self.pe_c_inds),1] )
        self.pe_c_inds_c = numpy.reshape( self.pe_c_inds, [1,len(self.pe_c_inds)] )

        self.pa_inds = range( c_end+p.Na+p.Nc+p.N, c_end+p.Na+p.Nc+p.N +p.Na )
        self.pa_inds_r = numpy.reshape( self.pa_inds, [len(self.pa_inds),1] )
        self.pa_inds_c = numpy.reshape( self.pa_inds, [1,len(self.pa_inds)] )

        self.pc_inds = range( c_end+p.Na+p.Nc+p.N+p.Na, c_end+p.Na+p.Nc+p.N+p.Na +p.Nc )
        self.pc_inds_r = numpy.reshape( self.pc_inds, [len(self.pc_inds),1] )
        self.pc_inds_c = numpy.reshape( self.pc_inds, [1,len(self.pc_inds)] )

        # second set for manual jac version
        c_end = 0
        self.ja_inds2 = range(c_end, c_end+p.Na)
        self.ja_inds_r2 = numpy.reshape( self.ja_inds2, [len(self.ja_inds2),1] )
        self.ja_inds_c2 = numpy.reshape( self.ja_inds2, [1,len(self.ja_inds2)] )

        self.jc_inds2 = range(c_end+p.Na, c_end+p.Na +p.Nc)
        self.jc_inds_r2 = numpy.reshape( self.jc_inds2, [len(self.jc_inds2),1] )
        self.jc_inds_c2 = numpy.reshape( self.jc_inds2, [1,len(self.jc_inds2)] )
        
        self.pe_inds2   = range( c_end+p.Na+p.Nc, c_end+p.Na+p.Nc +p.N )
        self.pe_inds_r2 = numpy.reshape( self.pe_inds2, [len(self.pe_inds2),1] )
        self.pe_inds_c2 = numpy.reshape( self.pe_inds2, [1,len(self.pe_inds2)] )

        self.pe_a_inds2 = range( c_end+p.Na+p.Nc, c_end+p.Na+p.Nc +p.Na )
        self.pe_a_inds_r2 = numpy.reshape( self.pe_a_inds2, [len(self.pe_a_inds2),1] )
        self.pe_a_inds_c2 = numpy.reshape( self.pe_a_inds2, [1,len(self.pe_a_inds2)] )

        self.pe_c_inds2 = range( c_end+p.Na+p.Nc +p.Na+p.Ns, c_end+p.Na+p.Nc +p.N )
        self.pe_c_inds_r2 = numpy.reshape( self.pe_c_inds2, [len(self.pe_c_inds2),1] )
        self.pe_c_inds_c2 = numpy.reshape( self.pe_c_inds2, [1,len(self.pe_c_inds2)] )

        self.pa_inds2 = range( c_end+p.Na+p.Nc+p.N, c_end+p.Na+p.Nc+p.N +p.Na )
        self.pa_inds_r2 = numpy.reshape( self.pa_inds2, [len(self.pa_inds2),1] )
        self.pa_inds_c2 = numpy.reshape( self.pa_inds2, [1,len(self.pa_inds2)] )

        self.pc_inds2 = range( c_end+p.Na+p.Nc+p.N+p.Na, c_end+p.Na+p.Nc+p.N+p.Na +p.Nc )
        self.pc_inds_r2 = numpy.reshape( self.pc_inds2, [len(self.pc_inds2),1] )
        self.pc_inds_c2 = numpy.reshape( self.pc_inds2, [1,len(self.pc_inds2)] )


        ### Matrices for thermal calcs (gradient operators)
        self.Ga, self.Gc, self.G = grad_mat( p.Na, p.x_m_a ), grad_mat( p.Nc, p.x_m_c ), grad_mat( p.N, p.x_m )

        # Initialize the C arrays for the heat generation (these are useful for the Jacobian)
        junkQ = self.calc_heat( y0, numpy.zeros(p.Na), numpy.zeros(p.Nc), p.uref_a( y0[self.csa_inds[:p.Na]]/p.csa_max ), p.uref_c( y0[self.csc_inds[:p.Nc]]/p.csc_max ) )

        # Kinetic C array (also useful for the Jacobian)
        csa_ss = y0[self.csa_inds[:p.Na]]
        csc_ss = y0[self.csc_inds[:p.Nc]]
        ce = y0[self.ce_inds]
        T  = y0[self.T_ind]
        self.C_ioa = 2.0*p.ioa_interp(csa_ss/p.csa_max, T, grid=False).flatten()/p.F * numpy.sqrt( ce[:p.Na ]/p.ce_nom * (1.0 - csa_ss/p.csa_max) * (csa_ss/p.csa_max) )
        self.C_ioc = 2.0*p.ioc_interp(csc_ss/p.csc_max, T, grid=False).flatten()/p.F * numpy.sqrt( ce[-p.Nc:]/p.ce_nom * (1.0 - csc_ss/p.csc_max) * (csc_ss/p.csc_max) )

#        self.C_ioa = (2.0*self.io_a/self.F) * numpy.ones_like( csa_ss )
#        self.C_ioc = (2.0*self.io_a/self.F) * numpy.ones_like( csc_ss )

    def setup_model(self,y0, yd0, name) :

        Implicit_Problem.__init__(self,y0=y0,yd0=yd0,name=name)


    def phie_mats(self,) :
        """
        Electrolyte constant B_ce matrix
        """
        p = self.p

        Ba = [ (1.-p.t_plus)*asa/ea for ea, asa in zip(p.eps_a_vec,p.as_a) ]
        Bs = [  0.0                for i in range(p.Ns) ]
        Bc = [ (1.-p.t_plus)*asc/ec for ec, asc in zip(p.eps_c_vec,p.as_c) ]

        self.B_ce = numpy.diag( numpy.array(Ba+Bs+Bc, dtype='d') )

        Bap = [ asa*p.F for asa in p.as_a  ]
        Bsp = [   0.0 for i   in range(p.Ns) ]
        Bcp = [ asc*p.F for asc in p.as_c  ]

        self.B2_pe = numpy.diag( numpy.array(Bap+Bsp+Bcp, dtype='d') )

    def phis_mats(self,) :
        """
        Solid phase parameters and j vector matrices
        """
        p = self.p

        self.A_ps_a = flux_mat_builder( p.Na, p.x_m_a, numpy.ones_like(p.vols_a), p.sig_a_eff )
        self.A_ps_c = flux_mat_builder( p.Nc, p.x_m_c, numpy.ones_like(p.vols_c), p.sig_c_eff )

        # Grounding form for BCs (was only needed during testing, before BVK was incorporated for coupling
#        self.A_ps_a[-1,-1] = 2*self.A_ps_a[-1,-1]
#        self.A_ps_c[ 0, 0] = 2*self.A_ps_c[ 0, 0]

        Baps = numpy.array( [ asa*p.F*dxa for asa,dxa in zip(p.as_a, p.vols_a) ], dtype='d' )
        Bcps = numpy.array( [ asc*p.F*dxc for asc,dxc in zip(p.as_c, p.vols_c) ], dtype='d' )

        self.B_ps_a = numpy.diag( Baps )
        self.B_ps_c = numpy.diag( Bcps )

        self.B2_ps_a = numpy.zeros( p.Na, dtype='d' )
        self.B2_ps_a[ 0] = -1.
        self.B2_ps_c = numpy.zeros( p.Nc, dtype='d' )
        self.B2_ps_c[-1] = -1.

    def cs_mats(self,) :
        """
        Solid phase diffusion model
        """
        p = self.p

        ## 1D spherical diffusion model
        # A_cs pre build
        self.A_csa_single = flux_mat_builder( p.Nra, p.r_m_a, p.vols_ra_m, p.Dsa*(p.r_e_a**2) )
        self.A_csc_single = flux_mat_builder( p.Nrc, p.r_m_c, p.vols_rc_m, p.Dsc*(p.r_e_c**2) )

        # A_cs build up to the stacked full cs size (Nr and Nx)
        b = [self.A_csa_single]*p.Na
        self.A_cs_a = scipy.linalg.block_diag( *b )
        b = [self.A_csc_single]*p.Nc
        self.A_cs_c = scipy.linalg.block_diag( *b )

        # B_cs and C_cs are constant (i.e., are not state-dependent)
        self.B_csa_single = numpy.array( [ 0. for i in range(p.Nra-1) ]+[-1.*p.r_e_a[-1]**2/p.vols_ra_m[-1]], dtype='d' )
        self.B_csc_single = numpy.array( [ 0. for i in range(p.Nrc-1) ]+[-1.*p.r_e_c[-1]**2/p.vols_rc_m[-1]], dtype='d' )

        b = [self.B_csa_single]*p.Na
        self.B_cs_a = scipy.linalg.block_diag( *b ).T
        b = [self.B_csc_single]*p.Nc
        self.B_cs_c = scipy.linalg.block_diag( *b ).T

        # Particle surface concentration
        h_na  = p.r_e_a[-1] - p.r_m_a[-1]
        h_n1a = p.r_m_a[-1] - p.r_m_a[-2]

        h_nc  = p.r_e_c[-1] - p.r_m_c[-1]
        h_n1c = p.r_m_c[-1] - p.r_m_c[-2]

        self.a_n_a, self.b_n_a, self.c_n_a = right_side_coeffs( h_na, h_n1a )
        self.a_n_c, self.b_n_c, self.c_n_c = right_side_coeffs( h_nc, h_n1c )

        self.C_cs_a_single = numpy.array( [0. for i in range(p.Nra-2)]+[-self.a_n_a/self.c_n_a, -self.b_n_a/self.c_n_a], dtype='d' )
        self.C_cs_c_single = numpy.array( [0. for i in range(p.Nrc-2)]+[-self.a_n_c/self.c_n_c, -self.b_n_c/self.c_n_c], dtype='d' )

        self.C_cs_a = scipy.linalg.block_diag( *[self.C_cs_a_single]*p.Na )
        self.C_cs_c = scipy.linalg.block_diag( *[self.C_cs_c_single]*p.Nc )

        self.C_cs_a_avg = scipy.linalg.block_diag( *[1./((1./3.)*p.Rp_a**3)*p.vols_ra_m]*p.Na )
        self.C_cs_c_avg = scipy.linalg.block_diag( *[1./((1./3.)*p.Rp_c**3)*p.vols_rc_m]*p.Nc )

        self.C_cs_a_mean = 1./p.La*p.vols_a.dot(self.C_cs_a_avg)
        self.C_cs_c_mean = 1./p.Lc*p.vols_c.dot(self.C_cs_c_avg)

        # Particle core concentration
        h_na  = p.r_e_a[0] - p.r_m_a[0]
        h_n1a = p.r_m_a[1] - p.r_m_a[0]

        h_nc  = p.r_e_c[0] - p.r_m_c[0]
        h_n1c = p.r_m_c[1] - p.r_m_c[0]

        a_n_a, b_n_a, c_n_a = left_side_coeffs( h_na, h_n1a )
        a_n_c, b_n_c, c_n_c = left_side_coeffs( h_nc, h_n1c )

        C_cso_a_single = numpy.array( [-b_n_a/a_n_a, -c_n_a/a_n_a] + [0. for i in range(p.Nra-2)], dtype='d' )
        C_cso_c_single = numpy.array( [-b_n_c/a_n_c, -c_n_c/a_n_c] + [0. for i in range(p.Nrc-2)], dtype='d' )

        self.C_cso_a = scipy.linalg.block_diag( *[C_cso_a_single]*p.Na )
        self.C_cso_c = scipy.linalg.block_diag( *[C_cso_c_single]*p.Nc )

        # D_cs prelim values, note this is Ds(cs) dependent and therefore requires updating for state dependent Ds
        self.D_cs_a = -1.0/(p.Dsa*self.c_n_a)*numpy.eye( p.Na )
        self.D_cs_c = -1.0/(p.Dsc*self.c_n_c)*numpy.eye( p.Nc )

    def set_iapp( self, I_app ) :
        self.i_app = I_app / self.Ac


    # cs mats
    def update_cs_mats( self, csa, csc, csa_ss, csc_ss, csa_o, csc_o ) :

        p = self.p

        Acsa_list = [ [] for i in range(p.Na) ]
        Acsc_list = [ [] for i in range(p.Nc) ]

        Dsa_ss = [ 0. for i in range(p.Na) ]
        Dsc_ss = [ 0. for i in range(p.Nc) ]

        for ia in range(p.Na) :

            csa_m = csa[ia*p.Nra:(ia+1)*p.Nra]
            csa_e = numpy.array( [csa_o[ia]] + [ 0.5*(csa_m[i+1]+csa_m[i]) for i in range(p.Nra-1) ] + [csa_ss[ia]] )
            Ua_e  = p.uref_a( csa_e/p.csa_max )
            Dsa_e = p.Dsa_intp( Ua_e )

            Acsa_list[ia] = flux_mat_builder( p.Nra, p.r_m_a, p.vols_ra_m, Dsa_e*(p.r_e_a**2) )

            Dsa_ss[ia] = Dsa_e[-1]

        for ic in range(p.Nc) :

            csc_m = csc[ic*p.Nrc:(ic+1)*p.Nrc]
            csc_e = numpy.array( [csc_o[ic]] + [ 0.5*(csc_m[i+1]+csc_m[i]) for i in range(p.Nrc-1) ] + [csc_ss[ic]] )
            Uc_e  = p.uref_c( csc_e/p.csc_max )
            Dsc_e = p.Dsc_intp( Uc_e )

            Acsc_list[ic] = flux_mat_builder( p.Nrc, p.r_m_c, p.vols_rc_m, Dsc_e*(p.r_e_c**2) )

            Dsc_ss[ic] = Dsc_e[-1]

    #        b = self.A_csa_single.reshape(1,Nra,Nra).repeat(Na,axis=0)
        self.A_cs_a = scipy.linalg.block_diag( *Acsa_list )
        self.A_cs_c = scipy.linalg.block_diag( *Acsc_list )

        self.D_cs_a = numpy.diag( -1.0/(numpy.array(Dsa_ss)*self.c_n_a) )
        self.D_cs_c = numpy.diag( -1.0/(numpy.array(Dsc_ss)*self.c_n_c) )


    ## Define c_e functions
    def build_Ace_mat( self, c, T ) :

        p = self.p

        D_eff = self.Diff_ce( c, T )

        A = p.K_m.dot( flux_mat_builder( p.N, p.x_m, p.vols, D_eff ) )

        return A

    def Diff_ce( self, c, T, mid_on=0, eps_off=0 ) :

        p = self.p

        D_ce = p.De_intp( c, T, grid=False ).flatten()
        
        if eps_off :
            D_mid = D_ce
        else :
            D_mid = D_ce * p.eps_eff

        if mid_on :
            D_out = D_mid
        else :
            if type(c) == float :
                D_out = D_mid
            else :
                D_out = mid_to_edge( D_mid, p.x_e )

        return D_out

    ## Define phi_e functions
    def build_Ape_mat( self, c, T ) :
        
        p = self.p

        k_eff = self.kapp_ce( c, T )

        A = flux_mat_builder( p.N, p.x_m, p.vols, k_eff )

        A[-1,-1] = 2*A[-1,-1] # BC update for phi_e = 0

        return A

    def build_Bpe_mat( self, c, T ) :

        p = self.p

        gam = 2.*(1.-p.t_plus)*p.R_gas*T / p.F

        k_eff = self.kapp_ce( c, T )

        c_edge = mid_to_edge( c, p.x_e )

        B1 = flux_mat_builder( p.N, p.x_m, p.vols, k_eff*gam/c_edge )

        return B1

    def kapp_ce( self, c, T, mid_on=0, eps_off=0 ) :

        p = self.p

        k_ce = 1e-1*p.ke_intp( c, T, grid=False ).flatten() # 1e-1 converts from mS/cm to S/m (model uses SI units)

        if eps_off :
            k_mid = k_ce
        else :
            k_mid = k_ce * p.eps_eff

        if mid_on :
            k_out = k_mid
        else :
            if type(c) == float :
                k_out = k_mid
            else :
                k_out = mid_to_edge( k_mid, p.x_e )

        return k_out

    def build_Bjac_mat( self, eta, a, b ) :
            
        d = a*numpy.cosh( b*eta )*b

        return numpy.diag( d )

    def build_BjT_mat( self, T, a, b ) :
            
        d = a*numpy.cosh( b/T )*(-b/T**2)

        return d
        
    def get_voltage( self, y ) :
        """
        Return the cell potential
        """
        pc = y[self.pc_inds]
        pa = y[self.pa_inds]

        Vcell = pc[-1] - pa[0]

        return Vcell
        
    def get_eta_uref( self, csa, csc, ja_rxn, jc_rxn, phi_s_a, phi_s_c, phi ) :

        p = self.p

        csa_ss = (self.C_cs_a.dot(csa)).flatten() + (self.D_cs_a.dot(ja_rxn)).flatten()
        csc_ss = (self.C_cs_c.dot(csc)).flatten() + (self.D_cs_c.dot(jc_rxn)).flatten()

        Uref_a = p.uref_a( csa_ss/p.csa_max ) # anode   equilibrium potential
        Uref_c = p.uref_c( csc_ss/p.csc_max ) # cathode equilibrium potential

        eta_a  = phi_s_a - phi[:p.Na]  - Uref_a  # anode   overpotential
        eta_c  = phi_s_c - phi[-p.Nc:] - Uref_c  # cathode overpotential

        return eta_a, eta_c, Uref_a, Uref_c, csa_ss, csc_ss

    def update_Cio( self, csa_ss, csc_ss, ce, T ) :

        p = self.p

        self.C_ioa = 2.0*p.ioa_interp(csa_ss/p.csa_max, T, grid=False).flatten()/p.F * numpy.sqrt( ce[:p.Na ]/p.ce_nom * (1.0 - csa_ss/p.csa_max) * (csa_ss/p.csa_max) )
        self.C_ioc = 2.0*p.ioc_interp(csc_ss/p.csc_max, T, grid=False).flatten()/p.F * numpy.sqrt( ce[-p.Nc:]/p.ce_nom * (1.0 - csc_ss/p.csc_max) * (csc_ss/p.csc_max) )

    def calc_heat( self, y, eta_a, eta_c, Uref_a, Uref_c ) :
        """
        Return the total integrated heat source across the cell sandwich
        """
        p = self.p

        ce      = y[ self.ce_inds  ]
        csa     = y[ self.csa_inds ]
        csc     = y[ self.csc_inds ]
        ja      = y[ self.ja_inds  ]
        jc      = y[ self.jc_inds  ]
        phi     = y[ self.pe_inds  ]
        phi_s_a = y[ self.pa_inds  ]
        phi_s_c = y[ self.pc_inds  ]
        T = y[self.T_ind]

        # Gradients for heat calc
        dphi_s_a = numpy.gradient( phi_s_a ) / numpy.gradient( p.x_m_a )
        dphi_s_c = numpy.gradient( phi_s_c ) / numpy.gradient( p.x_m_c )

        dphi = numpy.gradient( phi ) / numpy.gradient( p.x_m )

        dlnce = 1./ce * ( numpy.gradient(ce) / numpy.gradient( p.x_m ) )

        kapp_eff_m = self.kapp_ce( ce, T, mid_on=1 ) # kapp_eff at the node points (middle of control volume, rather than edge)

        # Reaction kinetics heat
        C_ra = (p.vols_a*p.F*p.as_a)
        C_rc = (p.vols_c*p.F*p.as_c)

        Q_rxn_a = C_ra.dot( ja*eta_a )
        Q_rxn_c = C_rc.dot( jc*eta_c )
        Q_rxn = Q_rxn_a + Q_rxn_c

        csa_mean = self.C_cs_a_avg.dot(csa)
        csc_mean = self.C_cs_c_avg.dot(csc)
        Uam = p.uref_a( csa_mean/p.csa_max )
        Ucm = p.uref_c( csc_mean/p.csc_max )

        eta_conc_a = Uref_a-Uam
        eta_conc_c = Uref_c-Ucm

        Q_conc_a = C_ra.dot( eta_conc_a*ja )
        Q_conc_c = C_rc.dot( eta_conc_c*jc )

        Q_conc = Q_conc_a + Q_conc_c

        # Ohmic heat in electrolyte and solid
        C_pe = (p.vols.dot( numpy.diag(kapp_eff_m*dphi).dot(self.G) ) + 
                p.vols.dot( numpy.diag(2*kapp_eff_m*p.R_gas*T/p.F*(1.-p.t_plus)*dlnce).dot(self.G) ))

        Q_ohm_e = C_pe.dot(phi)

        C_pa = p.vols_a.dot( numpy.diag(p.sig_a_eff*dphi_s_a).dot(self.Ga) )
        C_pc = p.vols_c.dot( numpy.diag(p.sig_c_eff*dphi_s_c).dot(self.Gc) )

        Q_ohm_s = C_pa.dot(phi_s_a) + C_pc.dot(phi_s_c)

        Q_ohm = Q_ohm_e + Q_ohm_s

        # Entropic heat
        ## ??

        # Total heat
        Q_tot = Q_ohm + Q_rxn + Q_conc

        self.C_q_pe = C_pe
        self.C_q_pa = C_pa
        self.C_q_pc = C_pc
        self.C_q_na = C_ra*ja
        self.C_q_nc = C_rc*jc
        self.C_q_ja = C_ra*eta_a + C_ra*eta_conc_a
        self.C_q_jc = C_rc*eta_c + C_rc*eta_conc_c

        return Q_tot

    ## Define system equations
    def res( self, t, y, yd ) :

        p = self.p

        ## Parse out the states
        # E-lyte conc
        ce     = y[ self.ce_inds]
        c_dots = yd[self.ce_inds]

        # Solid conc a:anode, c:cathode
        csa    = y[ self.csa_inds]
        csc    = y[ self.csc_inds]
        csa_dt = yd[self.csa_inds]
        csc_dt = yd[self.csc_inds]

        # Reaction (Butler-Volmer Kinetics)
        ja_rxn = y[self.ja_inds]
        jc_rxn = y[self.jc_inds]

        # E-lyte potential
        phi = y[self.pe_inds]

        # Solid potential
        phi_s_a = y[self.pa_inds]
        phi_s_c = y[self.pc_inds]

        # Thermal
        T    = y[ self.T_ind]
        T_dt = yd[self.T_ind]

        ## Grab state dependent matrices
        # For E-lyte conc and potential (i.e., De(ce), kapp_e(ce))
        A_ce = self.build_Ace_mat( ce, T )
        A_pe = self.build_Ape_mat( ce, T )
        B_pe = self.build_Bpe_mat( ce, T )

        ## Compute extra variables
        # For the reaction kinetics
        eta_a, eta_c, Uref_a, Uref_c, csa_ss, csc_ss = self.get_eta_uref( csa, csc, ja_rxn, jc_rxn, phi_s_a, phi_s_c, phi )

        # For Solid conc Ds
        csa_o = (self.C_cso_a.dot(csa)).flatten()
        csc_o = (self.C_cso_c.dot(csc)).flatten()

        self.update_cs_mats( csa, csc, csa_ss, csc_ss, csa_o, csc_o )

        # For kinetics, the io param is now conc dependent
        self.update_Cio( csa_ss, csc_ss, ce, T )

        Q_in = self.calc_heat( y, eta_a, eta_c, Uref_a, Uref_c )

        Q_out = p.h*p.Aconv*(T - self.T_amb)

        ja = self.C_ioa * numpy.sinh( 0.5*p.F/(p.R_gas*T)*eta_a )
        jc = self.C_ioc * numpy.sinh( 0.5*p.F/(p.R_gas*T)*eta_c )

        j = numpy.concatenate( [ ja_rxn, numpy.zeros(p.Ns), jc_rxn ] )

        ## Compute the residuals
        # Time deriv components
        r1 = c_dots - ( ((A_ce.dot(ce)).flatten() + (self.B_ce.dot(j)).flatten()) ) # E-lyte conc

        r2 = csa_dt - (self.A_cs_a.dot(csa).flatten() + self.B_cs_a.dot(ja_rxn).flatten()) # Anode   conc
        r3 = csc_dt - (self.A_cs_c.dot(csc).flatten() + self.B_cs_c.dot(jc_rxn).flatten()) # Cathode conc

        r4 = T_dt - 1./(p.rho*p.Cp)*(Q_in - Q_out)

        # Algebraic components
        r5 = ja_rxn - ja
        r6 = jc_rxn - jc 

        r7 = A_pe.dot(phi).flatten() - B_pe.dot(ce).flatten() + self.B2_pe.dot(j).flatten() # E-lyte potential

        r8 = self.A_ps_a.dot(phi_s_a).flatten() - self.B_ps_a.dot(ja_rxn).flatten() - self.B2_ps_a*self.i_app # Anode   potential
        r9 = self.A_ps_c.dot(phi_s_c).flatten() - self.B_ps_c.dot(jc_rxn).flatten() + self.B2_ps_c*self.i_app # Cathode potential

        res_out = numpy.concatenate( [r1, r2, r3, [r4], r5, r6, r7, r8, r9] )

        return res_out

    def jac( self, c, t, y, yd ) :

        p = self.p

        ### Setup 
        ## Parse out the states
        # E-lyte conc
        ce     = y[ self.ce_inds]

        # Solid conc a:anode, c:cathode
        csa    = y[ self.csa_inds]
        csc    = y[ self.csc_inds]

        # Reaction (Butler-Volmer Kinetics)
        ja_rxn = y[self.ja_inds]
        jc_rxn = y[self.jc_inds]

        # E-lyte potential
        phi = y[self.pe_inds]

        # Solid potential
        phi_s_a = y[self.pa_inds]
        phi_s_c = y[self.pc_inds]

        # Temp
        T = y[self.T_ind]

        ## Grab state dependent matrices
        # For E-lyte conc and potential (i.e., De(ce), kapp_e(ce))
        A_ce = self.build_Ace_mat( ce, T )
        A_pe = self.build_Ape_mat( ce, T )
        B_pe = self.build_Bpe_mat( ce, T )

        ## Compute extra variables
        # For the reaction kinetics
        eta_a, eta_c, Uref_a, Uref_c, csa_ss, csc_ss = self.get_eta_uref( csa, csc, ja_rxn, jc_rxn, phi_s_a, phi_s_c, phi )

        ### Build the Jac matrix
        ## Self coupling
        A_dots = numpy.diag( [1*c for i in range(p.num_diff_vars)] )
        j_c    = A_dots - scipy.linalg.block_diag( A_ce, self.A_cs_a, self.A_cs_c, [-p.h*p.Aconv/p.rho/p.Cp] )

        Bjac_a = self.build_Bjac_mat( eta_a, self.C_ioa, 0.5*p.F/(p.R_gas*T) )
        Bjac_c = self.build_Bjac_mat( eta_c, self.C_ioc, 0.5*p.F/(p.R_gas*T) )

        BjT_a = self.build_BjT_mat( T, self.C_ioa, 0.5*p.F/(p.R_gas)*eta_a )
        BjT_c = self.build_BjT_mat( T, self.C_ioc, 0.5*p.F/(p.R_gas)*eta_c )

#        dcss_dcs_a = self.C_cs_a_single
#        dcss_dcs_c = self.C_cs_c_single

        dcss_dja = numpy.diagonal( self.D_cs_a )
        dcss_djc = numpy.diagonal( self.D_cs_c )

        dU_csa_ss = (1.0/p.csa_max)*p.duref_a(csa_ss/p.csa_max)
        dU_csc_ss = (1.0/p.csc_max)*p.duref_c(csc_ss/p.csc_max)

        DUDcsa_ss = numpy.diag( dU_csa_ss )
        DUDcsc_ss = numpy.diag( dU_csc_ss )

        A_ja = numpy.diag(numpy.ones(p.Na)) - (Bjac_a.dot(-1.0*DUDcsa_ss*1.0)).dot( self.D_cs_a )
        A_jc = numpy.diag(numpy.ones(p.Nc)) - (Bjac_c.dot(-1.0*DUDcsc_ss*1.0)).dot( self.D_cs_c )

        j = scipy.linalg.block_diag( j_c, A_ja, A_jc, A_pe, self.A_ps_a, self.A_ps_c )

        ## Cross coupling
        # c_e: j coupling back in
        j[ numpy.ix_(self.ce_inds, self.ja_inds) ] = -self.B_ce[:, :p.Na ]
        j[ numpy.ix_(self.ce_inds, self.jc_inds) ] = -self.B_ce[:, -p.Nc:]

        # cs_a: j coupling
        j[ numpy.ix_(self.csa_inds, self.ja_inds) ] = -self.B_cs_a
        # cs_c: j coupling
        j[ numpy.ix_(self.csc_inds, self.jc_inds) ] = -self.B_cs_c

        a_coeff = 2.0*self.C_q_na*(-1.0)*dU_csa_ss
        Ca_T = numpy.array([ self.C_cs_a_single*ac for ac in a_coeff ]).flatten()
        c_coeff = 2.0*self.C_q_nc*(-1.0)*dU_csc_ss
        Cc_T = numpy.array([ self.C_cs_c_single*cc for cc in c_coeff ]).flatten()

        # T
        j[self.T_ind,self.ja_inds]  = -1./(p.rho*p.Cp)*(self.C_q_ja + 2.0*(self.C_q_na*(-1.0)*dU_csa_ss*dcss_dja))
        j[self.T_ind,self.jc_inds]  = -1./(p.rho*p.Cp)*(self.C_q_jc + 2.0*(self.C_q_nc*(-1.0)*dU_csc_ss*dcss_djc))
        j[self.T_ind,self.pe_inds]  = -1./(p.rho*p.Cp)*(self.C_q_pe + numpy.array( list(self.C_q_na)+[0. for i in range(p.Ns)]+list(self.C_q_nc) )*(-1.0))
        j[self.T_ind,self.pa_inds]  = -1./(p.rho*p.Cp)*(self.C_q_pa + self.C_q_na*(1.0))
        j[self.T_ind,self.pc_inds]  = -1./(p.rho*p.Cp)*(self.C_q_pc + self.C_q_nc*(1.0))
        j[self.T_ind,self.csa_inds] = -1./(p.rho*p.Cp)*(Ca_T)
        j[self.T_ind,self.csc_inds] = -1./(p.rho*p.Cp)*(Cc_T)

        j[self.ja_inds,self.T_ind] = -BjT_a
        j[self.jc_inds,self.T_ind] = -BjT_c


        # j_a: pe, pa, csa  coupling
        j[numpy.ix_(self.ja_inds, self.pa_inds  )] = -Bjac_a*( 1.0)
        j[numpy.ix_(self.ja_inds, self.pe_a_inds)] = -Bjac_a*(-1.0)
        j[numpy.ix_(self.ja_inds, self.csa_inds )] = -(Bjac_a.dot(-1.0*DUDcsa_ss*1.0)).dot( self.C_cs_a )

        # j_c: pe, pc, csc  coupling         
        j[numpy.ix_(self.jc_inds, self.pc_inds  )] = -Bjac_c*( 1.0)
        j[numpy.ix_(self.jc_inds, self.pe_c_inds)] = -Bjac_c*(-1.0)
        j[numpy.ix_(self.jc_inds, self.csc_inds )] = -(Bjac_c.dot(-1.0*DUDcsc_ss*1.0)).dot( self.C_cs_c )

        # phi_e: ce coupling into phi_e equation
        j[numpy.ix_(self.pe_inds,self.ce_inds)] = -B_pe
        j[numpy.ix_(self.pe_inds,self.ja_inds)] = self.B2_pe[:,:p.Na]
        j[numpy.ix_(self.pe_inds,self.jc_inds)] = self.B2_pe[:,-p.Nc:]

        # phi_s_a: ja
        j[numpy.ix_(self.pa_inds,self.ja_inds)] = -self.B_ps_a
        # phi_s_c: jc
        j[numpy.ix_(self.pc_inds,self.jc_inds)] = -self.B_ps_c
        ###        

        return j


class wraptest() :
    def __init__(self,conf_data) :
    
#        confdir    = '/home/mk-sim-linux/Battery_TempGrad/Python/batt_simulation/battsimpy/configs/'
#        
#        configFile = confdir+'model_fvmP2D.conf'
#        simFile    = confdir+'sim_DCR.conf'
#        
#        mcd = confreader.reader( configFile )
#        scd = confreader.reader( simFile )
#        
#        conf_data = mcd.conf_data.copy()
#        conf_data.update(scd.conf_data)    
    
        self.confDat = conf_data
        self.Pdat    = { 'RunInput':self.confDat }
        self.V_init  = 4.198 # [V]    
    
        self.p = params.params()
        self.p.buildpars( self.V_init, self.Pdat )    
    
        self.p.Ac = self.p.Area
    
        csa_max = 2.25*100**3/(96485.0/3.6/(365.0)) # [mol/m^3]
        csc_max = 4.74*100**3/(96485.0/3.6/(282.0)) # [mol/m^3]
        #csa_max = 30555 # [mol/m^3]
        #csc_max = 51554 # [mol/m^3]

        #bsp_dir = '/home/m_klein/Projects/battsimpy/'
        bsp_dir = '/home/mk-sim-linux/Battery_TempGrad/Python/batt_simulation/battsimpy/'
        #bsp_dir = '/Users/mk/Desktop/battsim/battsimpy/'

        #Ua_path = bsp_dir+'data/Model_v1/Model_Pars/solid/thermodynamics/uref_anode_bds.csv' #bigx.csv'
        #Uc_path = bsp_dir+'data/Model_v1/Model_Pars/solid/thermodynamics/uref_cathode_bds.csv' #bigx.csv'
        Ua_path = bsp_dir+'data/Model_v1/Model_Pars/solid/thermodynamics/Un_simba_20170512.csv'
        Uc_path = bsp_dir+'data/Model_v1/Model_Pars/solid/thermodynamics/Up_simba_20170512.csv'

        uref_a, uref_c, duref_a, duref_c = get_smooth_Uref_data( Ua_path, Uc_path, filter_on=0 )
        #uref_a, uref_c, duref_a, duref_c = get_smooth_Uref_data( Ua_path, Uc_path, ffa=0.4, ffc=0.2 )

        xa_init, xc_init = self.p.theta_a0, self.p.theta_c0 #0.743, 0.480 #0.767, 0.47
        ca_init = xa_init*csa_max 
        cc_init = xc_init*csc_max
        Ua_init = uref_a( xa_init )
        Uc_init = uref_c( xc_init )

        #print Ua_init
        #print Uc_init

        print 'Initial Voltage:', Uc_init - Ua_init

        ### Mesh
#        La = 68.0*1e-6
#        Ls = 25.0*1e-6
#        Lc = 52.0*1e-6
#        Lt = (La+Ls+Lc)
#        X = Lt # [m]
#
#        N  = 80
#        Ns = 10 #int(N*(Ls/Lt))
#        Na = 45 #int(N*(La/Lt))
#        Nc = N - Ns - Na
#
#        print 'Na, Ns, Nc:', Na, Ns, Nc
#
#        Nra = 10
#        Nrc = 20
#
#        Ra = 7.4e-6
#        Rc = 6.25e-6    

        ### Initial conditions
        # E-lyte conc
        c_init = 1000.0 # [mol/m^3]
        c_centered = c_init*numpy.ones( self.p.N, dtype='d' )
        # E-lyte potential
        p_init = 0.0 # [V]
        p_centered = p_init*numpy.ones( self.p.N, dtype='d' )
        # Solid potential on anode and cathode
        pa_init = Ua_init #0.0 # [V]
        pa_centered = pa_init*numpy.ones( self.p.Na, dtype='d' )
        pc_init = Uc_init #0.0 # [V]
        pc_centered = pc_init*numpy.ones( self.p.Nc, dtype='d' )
        # Solid conc on anode and cathode
        ca_centered = ca_init*numpy.ones( self.p.Na*self.p.Nra, dtype='d' )
        cc_centered = cc_init*numpy.ones( self.p.Nc*self.p.Nrc, dtype='d' )
        # j init
        ja = numpy.zeros(self.p.Na)
        jc = numpy.zeros(self.p.Nc)
        # T init
        T = 273.15+30.

        self.num_diff_vars = len(c_centered)+len(ca_centered)+len(cc_centered)+1
        self.num_algr_vars = len(ja)+len(jc)+len(p_centered)+len(pa_centered)+len(pc_centered)

        #The initial conditons
        y0  = numpy.concatenate( [c_centered, ca_centered, cc_centered, [T], ja, jc, p_centered, pa_centered, pc_centered] ) #Initial conditions
        yd0 = [0.0 for i in range(len(y0))] #Initial conditions

#        cell_coated_area = 1.0 # [m^2]


        # Initialize the parameters
#        self.p = params(Na,Ns,Nc,Nra,Nrc,La,Ls,Lc,Ra,Rc,cell_coated_area,bsp_dir)    
    
        imp_mod = MyProblem(self.p,y0,yd0)
        imp_mod.setup_model(y0,yd0,'test')
        imp_mod.p = self.p

        self.pars = self.p
        imp_mod.pars = self.p

        self.imp_mod = imp_mod        
        
    
    def simulate(self,Crate) :
        
        imp_mod = self.imp_mod        
        
        #Crate = 5./29.
#        cellname='uref_simba_0' #'val00_constElyte_epsa0p9_'
#        for Crate in [1.] :#[ 5./29., 0.5, 1., 2., 3., 4. ] :
        Vcut  = 2.95 # [V], cutoff voltage for end of dischargeF
        ce_lims = [1.,3990.]

        cell_cap = 29.0
        I_app = Crate*cell_cap # A
        #i_app = I_app / cell_coated_area # current density, [A/m^2]

        #Sets the options to the problem
        imp_mod.algvar = [1.0 for i in range(self.num_diff_vars)] + [0.0 for i in range(self.num_algr_vars)] #Set the algebraic components

        #Create an Assimulo implicit solver (IDA)
        imp_sim = IDA(imp_mod) #Create a IDA solver

        #Sets the paramters
        imp_sim.atol = 1e-5 #Default 1e-6
        imp_sim.rtol = 1e-5 #Default 1e-6
        imp_sim.suppress_alg = True #Suppres the algebraic variables on the error test

        imp_sim.display_progress = False
        imp_sim.verbosity = 50
        imp_sim.report_continuously = True
        imp_sim.time_limit = 10.

        #print dir(imp_sim)

    #    imp_sim.num_threads = 4
    #    print 'num_threads', imp_sim.num_threads

        ### Simulate
        t01, t02 = 0.01, 0.02

        imp_mod.set_iapp( I_app/10. )
        imp_sim.make_consistent('IDA_YA_YDP_INIT')
        ta, ya, yda = imp_sim.simulate(t01,2) 

        imp_mod.set_iapp( I_app/2. )
        imp_sim.make_consistent('IDA_YA_YDP_INIT')
        tb, yb, ydb = imp_sim.simulate(t02,2) 

        sim_on = 1
        if sim_on :
            # Sim step 1
            #imp_mod.set_iapp( I_app )
            #imp_sim.make_consistent('IDA_YA_YDP_INIT')
            #t1, y1, yd1 = imp_sim.simulate(1.0/Crate*3600.0,100) 

            NT = 80
            tfinal = 1.0/Crate*3600.0*1.2
            #time   = numpy.linspace( t02+0.1, 1.0/Crate*3600.0*31/29., NT )
            ti = tb
            delta_t = tfinal/NT
            t_out  = [] # 0 for ts in time ]
            V_out  = [] # 0 for ts in time ]
            y_out  = [] # numpy.zeros( [len(time), yb.shape[ 1]] )
            yd_out = [] #numpy.zeros( [len(time), ydb.shape[1]] )

            xa_bar = []
            xc_bar = []
            Ua_bar = []
            Uc_bar = []

            eta_a_out = []
            eta_c_out = []
            eta_full_x = []

            j_full_x = []
            Uss_full_x = []
            css_full_x = []
            phis_full_x = []

            ke_out = []
            De_out = []

            it = 0
            V_cell = imp_mod.get_voltage( yb[-1,:].flatten() )
            ce_now = yb[-1,imp_mod.ce_inds].flatten()
            print 'V_cell prior to time loop:', V_cell

            imp_mod.set_iapp( I_app )
            imp_sim.make_consistent('IDA_YA_YDP_INIT')
            sim_stopped = 0

            dV_tol = 0.02
            delta_t = 0.1
            refined_dt = 0

            while V_cell > Vcut and max(ce_now)<max(ce_lims) and min(ce_now)>min(ce_lims) and not sim_stopped and imp_sim.t<=tfinal :
                if it > 2 and not refined_dt :
                    dV = abs(V_out[-1] - V_out[-2])
                    delta_t = dV_tol/dV * delta_t

                try :
                    ti, yi, ydi = imp_sim.simulate(imp_sim.t+delta_t,2)

                except :
                    try :
                        if imp_sim.t > 0.8*tfinal :
                            delta_t = delta_t*.1
                            refined_dt = 1
                        ti, yi, ydi = imp_sim.simulate(imp_sim.t+delta_t*.01,3)
                        print '*** ran with refined delta_t ***'

                    except :
                        ti  = [t_out[it-1],t_out[it-1]]
                        yi  = numpy.array([ y_out[it-2], y_out[it-1] ])
                        ydi = numpy.array([ yd_out[it-2], yd_out[it-1] ])

                        sim_stopped = 1

                        print 'Sim stopped due time integration failure.'

                t_out.append( imp_sim.t )
                y_out.append( imp_sim.y )
                yd_out.append( imp_sim.yd )

                csa_now = yi[-1,imp_mod.csa_inds]
                csc_now = yi[-1,imp_mod.csc_inds]

                csa_mean = numpy.mean( imp_mod.C_cs_a_avg.dot(csa_now) )
                csc_mean = numpy.mean( imp_mod.C_cs_c_avg.dot(csc_now) )
                xa_mean  = csa_mean/imp_mod.p.csa_max
                xc_mean  = csc_mean/imp_mod.p.csc_max
                Uam = imp_mod.p.uref_a( xa_mean )
                Ucm = imp_mod.p.uref_c( xc_mean )

                xa_bar.append(xa_mean)
                xc_bar.append(xc_mean)
                Ua_bar.append(Uam)
                Uc_bar.append(Ucm)

                V_cell = imp_mod.get_voltage( imp_sim.y )

                V_out.append( V_cell )

                ce_now = imp_sim.y[imp_mod.ce_inds]

                ja_now = imp_sim.y[imp_mod.ja_inds]
                jc_now = imp_sim.y[imp_mod.jc_inds]
                pa_now = imp_sim.y[imp_mod.pa_inds]
                pc_now = imp_sim.y[imp_mod.pc_inds]
                pe_now = imp_sim.y[imp_mod.pe_inds]

                T_now  = imp_sim.y[imp_mod.T_ind]

                eta_a, eta_c, Uref_a_ss, Uref_c_ss, csa_ss, csc_ss = imp_mod.get_eta_uref( csa_now, csc_now, ja_now, jc_now, pa_now, pc_now, pe_now )

                eta_a_out.append(eta_a)
                eta_c_out.append(eta_c)

                eta_full_x.append(  numpy.concatenate( [eta_a ,    numpy.zeros(imp_mod.p.Ns), eta_c    ] ) )
                j_full_x.append(    numpy.concatenate( [ja_now,    numpy.zeros(imp_mod.p.Ns), jc_now   ] ) )
                Uss_full_x.append(  numpy.concatenate( [Uref_a_ss, numpy.zeros(imp_mod.p.Ns), Uref_c_ss] ) )
                phis_full_x.append( numpy.concatenate( [pa_now,    numpy.zeros(imp_mod.p.Ns), pc_now   ] ) )
                css_full_x.append( numpy.concatenate(  [csa_ss,    numpy.zeros(imp_mod.p.Ns), csc_ss   ] ) )

                ke_mid = imp_mod.kapp_ce( ce_now, T_now, mid_on=1, eps_off=1 )
                De_mid = imp_mod.Diff_ce( ce_now, T_now, mid_on=1, eps_off=1 )

                ke_out.append( ke_mid )
                De_out.append( De_mid )

                print 'time:',round(imp_sim.t,3), ' |  Voltage:', round(V_cell,3), ' |  '+str(round(imp_sim.t/tfinal*100.,1))+'% complete'

                if V_cell < Vcut :
                    print '\n','Vcut stopped simulation.'
                elif max(ce_now)>max(ce_lims) :
                    print '\n','ce max stopped simulation.'
                elif min(ce_now)<min(ce_lims) :
                    print '\n','ce min stopped simulation.'

                it+=1

            t1  = numpy.array(t_out)
            y1  = numpy.array(y_out)
            yd1 = numpy.array(yd_out)

            ce = y1[:,imp_mod.ce_inds]

            ce_1 = y1[:,imp_mod.ce_inds]
            pe_1 = y1[:,imp_mod.pe_inds]
            pa_1 = y1[:,imp_mod.pa_inds]
            pc_1 = y1[:,imp_mod.pc_inds]

            ja_1 = y1[:,imp_mod.ja_inds]
            jc_1 = y1[:,imp_mod.jc_inds]

            pa_cc = pa_1[:, 0]
            pc_cc = pc_1[:,-1]

            pe_midsep = pe_1[:,int((Nc+Ns+Na)/2.)]
            pe_mida   = pe_1[:,int(Na/2.)]
            pe_midc   = pe_1[:,Na+Ns+int(Nc/2.)]

            ce_midsep = ce_1[:,int((Nc+Ns+Na)/2.)]
            ce_mida   = ce_1[:,int(Na/2.)]
            ce_midc   = ce_1[:,Na+Ns+int(Nc/2.)]

            Vn = pa_cc - pe_midsep
            Vp = pc_cc - pe_midsep
#
            self.Vout = V_out
            self.tplot = t1            
            
#            tmin = numpy.array(t1)/60.
#
#            # save main outputs
#            numpy.savetxt( '../outdata/20170510/'+cellname+'_v_'+str(int(I_app))+'Adchg.csv', numpy.array([ tmin, t1, V_out, [numpy.amax(ce[it,:]) for it in range(len(ce[:,0]))], xa_bar, xc_bar, Ua_bar, Uc_bar, Vn, Vp, pa_cc, pc_cc, pe_mida, pe_midsep, pe_midc, ce_mida, ce_midsep, ce_midc]).T, delimiter=',',
#                            header='TestTime(min),TestTime(sec),Volts,Max ce,xa_bar,xc_bar,Ua_bar,Uc_bar,Vn,Vp,pa_cc,pc_cc,pe_mida,pe_midsep,pe_midc,ce_mida,ce_midsep,ce_midc' )
#
#            # save ce
#
#            ce_out = numpy.insert( ce_1, 0, t1, axis=1 )
#            ce_out = numpy.insert( ce_out, 0, [0]+list(imp_mod.x_m), axis=0 )
#
#            numpy.savetxt( '../outdata/20170510/'+cellname+'_ce_'+str(int(I_app))+'Adchg.csv', ce_out, delimiter=',' )
#
#            # save phi
#            pe_out = numpy.insert( pe_1, 0, t1, axis=1 )
#            pe_out = numpy.insert( pe_out, 0, [0]+list(imp_mod.x_m), axis=0 )
#
#            numpy.savetxt( '../outdata/20170510/'+cellname+'_pe_'+str(int(I_app))+'Adchg.csv', pe_out, delimiter=',' )
#
#            # save ke
#
#            ke_o = numpy.insert( ke_out, 0, t1, axis=1 )
#            ke_o = numpy.insert( ke_o, 0, [0]+list(imp_mod.x_m), axis=0 )
#
#            numpy.savetxt( '../outdata/20170510/'+cellname+'_kemid_'+str(int(I_app))+'Adchg.csv', ke_o, delimiter=',' )
#
#            # save De
#
#            De_o = numpy.insert( De_out, 0, t1, axis=1 )
#            De_o = numpy.insert( De_o, 0, [0]+list(imp_mod.x_m), axis=0 )
#
#            numpy.savetxt( '../outdata/20170510/'+cellname+'_Demid_'+str(int(I_app))+'Adchg.csv', De_o, delimiter=',' )
#
#            # save ja, jc
#            j_out = numpy.insert( j_full_x, 0, t1, axis=1 )
#            j_out = numpy.insert( j_out, 0, [0]+list(imp_mod.x_m), axis=0 )
#
#            numpy.savetxt( '../outdata/20170510/'+cellname+'_jrxn_fullx_'+str(int(I_app))+'Adchg.csv', j_out, delimiter=',' )
#
#            # save pa, pc
#            phis_out = numpy.insert( phis_full_x, 0, t1, axis=1 )
#            phis_out = numpy.insert( phis_out, 0, [0]+list(imp_mod.x_m), axis=0 )
#
#            numpy.savetxt( '../outdata/20170510/'+cellname+'_phis_fullx_'+str(int(I_app))+'Adchg.csv', phis_out, delimiter=',' )
#
#            # save eta_a, eta_c
#            eta_out = numpy.insert( eta_full_x, 0, t1, axis=1 )
#            eta_out = numpy.insert( eta_out, 0, [0]+list(imp_mod.x_m), axis=0 )
#
#            numpy.savetxt( '../outdata/20170510/'+cellname+'_eta_fullx_'+str(int(I_app))+'Adchg.csv', eta_out, delimiter=',' )
#
#            # save Uss_a, Uss_c
#            Uss_out = numpy.insert( Uss_full_x, 0, t1, axis=1 )
#            Uss_out = numpy.insert( Uss_out, 0, [0]+list(imp_mod.x_m), axis=0 )
#
#            numpy.savetxt( '../outdata/20170510/'+cellname+'_Uss_fullx_'+str(int(I_app))+'Adchg.csv', Uss_out, delimiter=',' )
#
#            # save css_a, css_c
#            css_out = numpy.insert( css_full_x, 0, t1, axis=1 )
#            css_out = numpy.insert( css_out, 0, [0]+list(imp_mod.x_m), axis=0 )
#
#            numpy.savetxt( '../outdata/20170510/'+cellname+'_css_fullx_'+str(int(I_app))+'Adchg.csv', css_out, delimiter=',' )


            plot_on = 0
            if plot_on :
                # Sim step 2
                imp_mod.set_iapp( 0.0 )
                imp_sim.make_consistent('IDA_YA_YDP_INIT')
                t2, y2, yd2 = imp_sim.simulate(imp_sim.t*1.3,20) 

                # extract variables
                im = imp_mod
                ce_1 = y1[:,im.ce_inds]
                ca_1 = y1[:,im.csa_inds]
                cc_1 = y1[:,im.csc_inds]

                ca1_r = [ numpy.reshape( ca_1[it,:], (im.Na, im.Nra) ) for it in range(len(t1)) ]
                cc1_r = [ numpy.reshape( cc_1[it,:], (im.Nc, im.Nrc) ) for it in range(len(t1)) ]

                T_1 = y1[:,im.T_ind]

                ce_2 = y2[:,im.ce_inds]
                ca_2 = y2[:,im.csa_inds]
                cc_2 = y2[:,im.csc_inds]

                ca2_r = [ numpy.reshape( ca_2[it,:], (im.Na, im.Nra) ) for it in range(len(t2)) ]
                cc2_r = [ numpy.reshape( cc_2[it,:], (im.Nc, im.Nrc) ) for it in range(len(t2)) ]

                pe_2 = y2[:,im.pe_inds]
                pa_2 = y2[:,im.pa_inds]
                pc_2 = y2[:,im.pc_inds]

                ja_2 = y2[:,im.ja_inds]
                jc_2 = y2[:,im.jc_inds]

                T_2 = y2[:,im.T_ind]

                #Plot
                # t1
                # Plot through space
                f, ax = plt.subplots(2,4)
                # ce vs x
                ax[0,0].plot(imp_mod.x_m*1e6,ce_1.T) 
                # pe vs x
                ax[0,1].plot(imp_mod.x_m*1e6,pe_1.T)
                # pa vs x
                ax[0,2].plot(imp_mod.x_m_a*1e6,pa_1.T)
                # pc vs x
                ax[0,2].plot(imp_mod.x_m_c*1e6,pc_1.T)

                ax[0,3].plot( t1, T_1 )

                ax[0,0].set_title('t1 c')
                ax[0,0].set_xlabel('Cell Thickness [$\mu$m]')
                ax[0,0].set_ylabel('E-lyte Conc. [mol/m$^3$]')
                ax[0,1].set_title('t1 p')
                ax[0,1].set_xlabel('Cell Thickness [$\mu$m]')
                ax[0,1].set_ylabel('E-lyte Potential [V]')
                ax[0,2].set_title('t1 p solid')
                ax[0,2].set_xlabel('Cell Thickness [$\mu$m]')
                ax[0,2].set_ylabel('Solid Potential [V]')
                #ax[0,3].set_title('t1 conc solid')
                #ax[0,3].set_xlabel('Cell Thickness [$\mu$m]')
                #ax[0,3].set_ylabel('Solid Conc. [mol/m$^3$]')

                # t2
                ax[1,0].plot(imp_mod.x_m*1e6,ce_2.T)
                ax[1,1].plot(imp_mod.x_m*1e6,pe_2.T)

                ax[1,2].plot(imp_mod.x_m_a*1e6,pa_2.T)
                ax[1,2].plot(imp_mod.x_m_c*1e6,pc_2.T)

                ax[1,3].plot( t2, T_2 )

                ax[1,0].set_title('t2 c')
                ax[1,0].set_xlabel('Cell Thickness [$\mu$m]')
                ax[1,0].set_ylabel('E-lyte Conc. [mol/m$^3$]')
                ax[1,1].set_title('t2 p e-lyte')
                ax[1,1].set_xlabel('Cell Thickness [$\mu$m]')
                ax[1,1].set_ylabel('E-lyte Potential [V]')
                ax[1,2].set_title('t2 p solid')
                ax[1,2].set_xlabel('Cell Thickness [$\mu$m]')
                ax[1,2].set_ylabel('Solid Potential [V]')
                #ax[1,3].set_title('t2 Solid Conc.')
                #ax[1,3].set_xlabel('Cell Thickness [$\mu$m]')
                #ax[1,3].set_ylabel('Solid Conc. [mol/m$^3$]')

                plt.tight_layout()

                fcs, ax = plt.subplots(1,2)
                ira, irc = im.Nra-1, im.Nrc-1
                for it in range(len(t1)) :
                    # ca vs x
                    ax[0].plot(imp_mod.x_m_a*1e6, ca1_r[it][:,ira]/imp_mod.csa_max)
                    # cc vs x
                    ax[0].plot(imp_mod.x_m_c*1e6, cc1_r[it][:,irc]/imp_mod.csc_max)


            #    for it in range(len(t1)) :
            #        ax[1].plot(imp_mod.x_m_a*1e6, ca2_r[it][:,ira]/imp_mod.csa_max)
            #        ax[1].plot(imp_mod.x_m_c*1e6, cc2_r[it][:,irc]/imp_mod.csc_max)

                ax[0].set_title('t1 Solid Conc.')
                ax[1].set_title('t2 Solid Conc.')
                ax[0].set_xlabel('Cell Thickness [$\mu$m]')
                ax[0].set_ylabel('Solid Conc. [mol/m$^3$]')

                plt.tight_layout()


                fcsr, ax = plt.subplots(1,2)
                ixa, ixc = im.Na-1, 0
                for it in range(len(t1)) :
                    # ca vs x
                    ax[0].plot(imp_mod.r_m_a*1e6, ca1_r[it][ixa,:])
                    # cc vs x
                    ax[0].plot(imp_mod.r_m_c*1e6, cc1_r[it][ixc,:])


            #    for it in range(len(t1)) :
            #        ax[1].plot(imp_mod.r_m_a*1e6, ca2_r[it][ixa,:])
            #        ax[1].plot(imp_mod.r_m_c*1e6, cc2_r[it][ixc,:])

                ax[0].set_title('t1 Solid Conc.')
                ax[1].set_title('t2 Solid Conc.')
                ax[0].set_xlabel('Cell Thickness [$\mu$m]')
                ax[0].set_ylabel('Solid Conc. [mol/m$^3$]')

                plt.tight_layout()


                # Plot through time
                f, ax = plt.subplots(1,3)
                ax[0].plot(t1,ce_1)
                ax[1].plot(t1,pe_1)
                ax[2].plot(t1,pa_1) 
                ax[2].plot(t1,pc_1)
                #ax[3].plot(t1,ca_1) 
                #ax[3].plot(t1,cc_1) 

                ax[0].plot(t2,ce_2)
                ax[1].plot(t2,pe_2)
                ax[2].plot(t2,pa_2) 
                ax[2].plot(t2,pc_2) 
                #ax[3].plot(t2,ca_2) 
                #ax[3].plot(t2,cc_2) 

                ax[0].set_ylabel('E-lyte Conc. [mol/m$^3$]')
                ax[0].set_xlabel('Time [s]')
                ax[1].set_ylabel('E-lyte Potential [V]')
                ax[1].set_xlabel('Time [s]')
                ax[2].set_ylabel('Solid Potential [V]')
                ax[2].set_xlabel('Time [s]')
            #    #ax[3].set_ylabel('Solid Conc. [mol/m$^3$]')
            #    #ax[3].set_xlabel('Time [s]')

                plt.tight_layout()

                plt.figure()
                plt.plot( t1, pc_1[:,-1] - pa_1[:,0] )
                plt.plot( t2, pc_2[:,-1] - pa_2[:,0] )

                plt.show()




#imp_mod = MyProblem(Na,Ns,Nc,Nra,Nrc,X,Ra,Rc,cell_coated_area,bsp_dir,y0,yd0,'Example using an analytic Jacobian')

## my own time solver

#delta_t = 1.0
#tf = 10.
#time = [ i*delta_t for i in range(int(tf/delta_t)+1) ]

#print time

#x_out = numpy.zeros( [num_diff_vars, len(time)] )
#z_out = numpy.zeros( [num_algr_vars, len(time)] )

#x_out[:,0] = numpy.concatenate( [c_centered, ca_centered, cc_centered, [T]] )
#z_out[:,0] = numpy.concatenate( [ja, jc, p_centered, pa_centered, pc_centered] )

#for it, t in enumerate(time[1:]) :

#    if it == 0 :
#        Cur_vec = [ 0.0, 0.0, 0.1*I_app ]
#    elif it == 1 :
#        Cur_vec = [ 0.0, 0.1*I_app, 0.5*I_app ]
#    elif it == 2 :
#        Cur_vec = [ 0.1*I_app, 0.5*I_app, I_app ]
#    elif it == 3 :
#        Cur_vec = [ 0.5*I_app, I_app, I_app ]
#    else :
#        Cur_vec = [ I_app, I_app, I_app ]
#        
#    x_out[:,it+1], z_out[:,it+1], newtonStats = imp_mod.cn_solver( x_out[:,it], z_out[:,it], Cur_vec, delta_t )

#plt.close()
#f, ax = plt.subplots(1,3)
#ax[0].plot( imp_mod.x_m, x_out[:imp_mod.N] )

#ax[1].plot( imp_mod.x_m, z_out[imp_mod.Na+imp_mod.Nc:imp_mod.Na+imp_mod.Nc+imp_mod.N,:-1] )

#ax[2].plot( imp_mod.x_m_a, z_out[-imp_mod.Na-imp_mod.Nc:-imp_mod.Nc,:-1] )
#ax[2].plot( imp_mod.x_m_c, z_out[-imp_mod.Nc:,:-1] )
#plt.show()

#print z_out




#    def dae_system( self, x, z, Input, get_mats=0 ) :

#        self.set_iapp( Input )

#        y = numpy.concatenate([x,z])

#        ## Parse out the states
#        # E-lyte conc
#        ce     = y[ self.ce_inds]

#        # Solid conc a:anode, c:cathode
#        csa    = y[ self.csa_inds]
#        csc    = y[ self.csc_inds]

#        # Reaction (Butler-Volmer Kinetics)
#        ja_rxn = y[self.ja_inds]
#        jc_rxn = y[self.jc_inds]

#        # E-lyte potential
#        phi = y[self.pe_inds]

#        # Solid potential
#        phi_s_a = y[self.pa_inds]
#        phi_s_c = y[self.pc_inds]

#        ## Grab state dependent matrices
#        # For E-lyte conc and potential (i.e., De(ce), kapp_e(ce))
#        A_ce = self.build_Ace_mat( ce )
#        A_pe = self.build_Ape_mat( ce )
#        B_pe = self.build_Bpe_mat( ce )

#        # For Solid conc Ds
#        csa_ss = (self.C_cs_a.dot(csa)).flatten() + (self.D_cs_a.dot(ja_rxn)).flatten()
#        csc_ss = (self.C_cs_c.dot(csc)).flatten() + (self.D_cs_c.dot(jc_rxn)).flatten()

#        csa_o = (self.C_cso_a.dot(csa)).flatten()
#        csc_o = (self.C_cso_c.dot(csc)).flatten()

#        self.update_cs_mats( csa, csc, csa_ss, csc_ss, csa_o, csc_o )

#        # Thermal
#        T = y[ self.T_ind]

#        ## Compute extra variables
#        # For the reaction kinetics
#        Uref_a = self.uref_a( csa_ss/self.csa_max ) # anode   equilibrium potential
#        Uref_c = self.uref_c( csc_ss/self.csc_max ) # cathode equilibrium potential

#        eta_a  = phi_s_a - phi[:self.Na]  - Uref_a  # anode   overpotential
#        eta_c  = phi_s_c - phi[-self.Nc:] - Uref_c  # cathode overpotential

#        Q_in = self.calc_heat( y, eta_a, eta_c )

#        Q_out = 1./(self.h*self.Aconv)*(T - self.T_amb)

##        ja = 2.0*self.io_a * numpy.sqrt( ce[:self.Na]/self.ce_nom * (1.0 - csa_ss/self.csa_max) * (csa_ss/self.csa_max) ) * numpy.sinh( self.R_gas/(2.0*self.F*self.T)*eta_a )
##        jc = 2.0*self.io_c * numpy.sqrt( ce[-self.Nc:]/self.ce_nom * (1.0 - csc_ss/self.csc_max) * (csc_ss/self.csc_max) ) * numpy.sinh( self.R_gas/(2.0*self.F*self.T)*eta_c )
#        ja = 2.0*self.io_a/self.F * numpy.sinh( 0.5*self.F/(self.R_gas*self.T)*eta_a )
#        jc = 2.0*self.io_c/self.F * numpy.sinh( 0.5*self.F/(self.R_gas*self.T)*eta_c )

#        j = numpy.concatenate( [ ja_rxn, numpy.zeros(self.Ns), jc_rxn ] )

#        ## Compute the residuals
#        # Time deriv components
#        r1 = ( ((A_ce.dot(ce)).flatten() + (self.B_ce.dot(j)).flatten()) ) # E-lyte conc

#        r2 = (self.A_cs_a.dot(csa).flatten() + self.B_cs_a.dot(ja_rxn).flatten()) # Anode   conc
#        r3 = (self.A_cs_c.dot(csc).flatten() + self.B_cs_c.dot(jc_rxn).flatten()) # Cathode conc

#        r4 = 1./(self.rho*self.Cp)*(Q_in - Q_out)

#        # Algebraic components
#        r5 = ja_rxn - ja
#        r6 = jc_rxn - jc 

#        r7 = A_pe.dot(phi).flatten() - B_pe.dot(ce).flatten() + self.B2_pe.dot(j).flatten() # E-lyte potential

#        r8 = self.A_ps_a.dot(phi_s_a).flatten() - self.B_ps_a.dot(ja_rxn).flatten() - self.B2_ps_a*self.i_app # Anode   potential
#        r9 = self.A_ps_c.dot(phi_s_c).flatten() - self.B_ps_c.dot(jc_rxn).flatten() + self.B2_ps_c*self.i_app # Cathode potential

#        if get_mats :
#            res_out = numpy.concatenate( [r1,r2,r3,[r4]] ), numpy.concatenate( [r5, r6, r7, r8, r9] ), { 'A_ce':A_ce, 'A_pe':A_pe, 'B_pe':B_pe, 'csa':csa, 'csc':csc, 'csa_ss':csa_ss, 'csc_ss':csc_ss, 'eta_a':eta_a, 'eta_c':eta_c }
#        else :
#            res_out = numpy.concatenate( [r1,r2,r3,[r4]] ), numpy.concatenate( [r5, r6, r7, r8, r9] )

#        return res_out

#    def dae_system_num( self, y ) :

#        ## Parse out the states
#        # E-lyte conc
#        ce     = y[ self.ce_inds]
#        # Solid conc a:anode, c:cathode
#        csa    = y[ self.csa_inds]
#        csc    = y[ self.csc_inds]

#        # Reaction (Butler-Volmer Kinetics)
#        ja_rxn = y[self.ja_inds]
#        jc_rxn = y[self.jc_inds]

#        # E-lyte potential
#        phi = y[self.pe_inds]

#        # Solid potential
#        phi_s_a = y[self.pa_inds]
#        phi_s_c = y[self.pc_inds]

#        ## Grab state dependent matrices
#        # For E-lyte conc and potential (i.e., De(ce), kapp_e(ce))
#        A_ce = self.build_Ace_mat( ce )
#        A_pe = self.build_Ape_mat( ce )
#        B_pe = self.build_Bpe_mat( ce )

#        # For Solid conc Ds
#        csa_ss = (self.C_cs_a.dot(csa)).flatten() + (self.D_cs_a.dot(ja_rxn)).flatten()
#        csc_ss = (self.C_cs_c.dot(csc)).flatten() + (self.D_cs_c.dot(jc_rxn)).flatten()

#        csa_o = (self.C_cso_a.dot(csa)).flatten()
#        csc_o = (self.C_cso_c.dot(csc)).flatten()

#        self.update_cs_mats( csa, csc, csa_ss, csc_ss, csa_o, csc_o )

#        # Thermal
#        T    = y[ self.T_ind]

#        ## Compute extra variables
#        # For the reaction kinetics
#        Uref_a = self.uref_a( csa_ss/self.csa_max ) # anode   equilibrium potential
#        Uref_c = self.uref_c( csc_ss/self.csc_max ) # cathode equilibrium potential

#        eta_a  = phi_s_a - phi[:self.Na]  - Uref_a  # anode   overpotential
#        eta_c  = phi_s_c - phi[-self.Nc:] - Uref_c  # cathode overpotential

#        Q_in = self.calc_heat( y, eta_a, eta_c )

#        Q_out = 1./(self.h*self.Aconv)*(T - self.T_amb)

##        ja = 2.0*self.io_a * numpy.sqrt( ce[:self.Na]/self.ce_nom * (1.0 - csa_ss/self.csa_max) * (csa_ss/self.csa_max) ) * numpy.sinh( self.R_gas/(2.0*self.F*self.T)*eta_a )
##        jc = 2.0*self.io_c * numpy.sqrt( ce[-self.Nc:]/self.ce_nom * (1.0 - csc_ss/self.csc_max) * (csc_ss/self.csc_max) ) * numpy.sinh( self.R_gas/(2.0*self.F*self.T)*eta_c )
#        ja = 2.0*self.io_a/self.F * numpy.sinh( 0.5*self.F/(self.R_gas*self.T)*eta_a )
#        jc = 2.0*self.io_c/self.F * numpy.sinh( 0.5*self.F/(self.R_gas*self.T)*eta_c )

#        j = numpy.concatenate( [ ja_rxn, numpy.zeros(self.Ns), jc_rxn ] )

#        ## Compute the residuals
#        # Time deriv components
#        r1 = ( ((A_ce.dot(ce)).flatten() + (self.B_ce.dot(j)).flatten()) ) # E-lyte conc

#        r2 = (self.A_cs_a.dot(csa).flatten() + self.B_cs_a.dot(ja_rxn).flatten()) # Anode   conc
#        r3 = (self.A_cs_c.dot(csc).flatten() + self.B_cs_c.dot(jc_rxn).flatten()) # Cathode conc

#        r4 = 1./(self.rho*self.Cp)*(Q_in - Q_out)

#        # Algebraic components
#        r5 = ja_rxn - ja
#        r6 = jc_rxn - jc 

#        r7 = A_pe.dot(phi).flatten() - B_pe.dot(ce).flatten() + self.B2_pe.dot(j).flatten() # E-lyte potential

#        r8 = self.A_ps_a.dot(phi_s_a).flatten() - self.B_ps_a.dot(ja_rxn).flatten() - self.B2_ps_a*self.i_app # Anode   potential
#        r9 = self.A_ps_c.dot(phi_s_c).flatten() - self.B_ps_c.dot(jc_rxn).flatten() + self.B2_ps_c*self.i_app # Cathode potential

#        res_out = numpy.concatenate( [r1,r2,r3,[r4], r5, r6, r7, r8, r9] )

#        return res_out


#    def jac_system( self, mats ) :

#        A_ce = mats['A_ce']
#        A_pe = mats['A_pe']
#        B_pe = mats['B_pe']

#        Bjac_a = self.build_Bjac_mat( mats['eta_a'], 2.0*self.io_a/self.F, 0.5*self.F/(self.R_gas*self.T) )
#        Bjac_c = self.build_Bjac_mat( mats['eta_c'], 2.0*self.io_c/self.F, 0.5*self.F/(self.R_gas*self.T) )

#        DUDcsa_ss = numpy.diag( (1.0/self.csa_max)*self.duref_a(mats['csa_ss']/self.csa_max) )
#        DUDcsc_ss = numpy.diag( (1.0/self.csc_max)*self.duref_c(mats['csc_ss']/self.csc_max) )

#        A_ja = numpy.diag(numpy.ones(self.Na)) - (Bjac_a.dot(-1.0*DUDcsa_ss*1.0)).dot( self.D_cs_a )
#        A_jc = numpy.diag(numpy.ones(self.Nc)) - (Bjac_c.dot(-1.0*DUDcsc_ss*1.0)).dot( self.D_cs_c )

#        ## fx
#        fx =  scipy.linalg.block_diag( A_ce, self.A_cs_a, self.A_cs_c, [-1./(self.rho*self.Cp)*1./(self.h*self.Aconv)] )
#        ##

#        ## fz
#        fz =  numpy.zeros( [self.num_diff_vars, self.num_algr_vars] )
#        # ce vs j
#        fz[ numpy.ix_(self.ce_inds, self.ja_inds2) ] = self.B_ce[:, :self.Na ]
#        fz[ numpy.ix_(self.ce_inds, self.jc_inds2) ] = self.B_ce[:, -self.Nc:]
#        # cs vs j
#        fz[ numpy.ix_(self.csa_inds, self.ja_inds2) ] = self.B_cs_a
#        fz[ numpy.ix_(self.csc_inds, self.jc_inds2) ] = self.B_cs_c
#        # T vs phi, phi_s_a, phi_s_c, ja, jc
#        fz[self.T_ind,self.ja_inds2] = 1./(self.rho*self.Cp)*self.C_q_ja
#        fz[self.T_ind,self.jc_inds2] = 1./(self.rho*self.Cp)*self.C_q_jc

#        fz[self.T_ind,self.pe_inds2] = 1./(self.rho*self.Cp)*(self.C_q_pe + numpy.array( list(self.C_q_na)+[0. for i in range(self.Ns)]+list(self.C_q_nc) )*(-1.0))

#        fz[self.T_ind,self.pa_inds2] = 1./(self.rho*self.Cp)*( self.C_q_pa + self.C_q_na*(1.0) )
#        fz[self.T_ind,self.pc_inds2] = 1./(self.rho*self.Cp)*( self.C_q_pc + self.C_q_nc*(1.0) )
#        ##

#        ## gx
#        gx =  numpy.zeros( [self.num_algr_vars, self.num_diff_vars] )
#        # j vs cs_ss
#        gx[ numpy.ix_(self.ja_inds2, self.csa_inds) ] = -(Bjac_a.dot(-1.0*DUDcsa_ss*1.0)).dot(self.C_cs_a)
#        gx[ numpy.ix_(self.jc_inds2, self.csc_inds) ] = -(Bjac_c.dot(-1.0*DUDcsc_ss*1.0)).dot(self.C_cs_c)
#        # phi_e vs ce
#        gx[ numpy.ix_(self.pe_inds2, self.ce_inds) ] = -B_pe
#        ##

#        ## gz
#        # z vs z
#        gz0 =  scipy.linalg.block_diag( A_ja, A_jc, A_pe, self.A_ps_a, self.A_ps_c )
#        # z cross coupling
#        gz00 = numpy.zeros_like( gz0 )
#        # phi_e vs j
#        gz00[ numpy.ix_(self.pe_inds2, self.ja_inds2) ] = self.B2_pe[:,:self.Na]
#        gz00[ numpy.ix_(self.pe_inds2, self.jc_inds2) ] = self.B2_pe[:,-self.Nc:]
#        # phi_s vs j
#        gz00[ numpy.ix_(self.pa_inds2, self.ja_inds2) ] = -self.B_ps_a
#        gz00[ numpy.ix_(self.pc_inds2, self.jc_inds2) ] = -self.B_ps_c
#        # j vs phi_s
#        gz00[ numpy.ix_(self.ja_inds2, self.pa_inds2) ] = -Bjac_a*( 1.0)
#        gz00[ numpy.ix_(self.jc_inds2, self.pc_inds2) ] = -Bjac_c*( 1.0)
#        # j vs phi_e
#        gz00[ numpy.ix_(self.ja_inds2, self.pe_a_inds2) ] = -Bjac_a*(-1.0)
#        gz00[ numpy.ix_(self.jc_inds2, self.pe_c_inds2) ] = -Bjac_c*(-1.0)

#        gz = gz0 + gz00

#        return fx, fz, gx, gz


#    def cn_solver( self, x, z, Cur_vec, delta_t ) :
#        """
#        Crank-Nicholson solver for marching through time
#        """
#        Cur_prev, Cur, Cur_nxt = Cur_vec[0], Cur_vec[1], Cur_vec[2]

#        maxIters = 20
#        tol      = 1e-5

#        Nx = self.num_diff_vars
#        Nz = self.num_algr_vars

#        x_nxt = numpy.zeros( (Nx,maxIters), dtype='d' )
#        z_nxt = numpy.zeros( (Nz,maxIters), dtype='d' )

#        relres = numpy.zeros( maxIters, dtype='d' )
#        relres[0] = 1.0

#        var_flag = {'lim_on':0}

#        # Solve for consistent ICs
#        if Cur != Cur_prev :    
#            z_cons = numpy.zeros( (Nz, maxIters), dtype='d' )
#            z_cons[:,0] = deepcopy(z)

#            junk_f, g, mats = self.dae_system( x, z, Cur, get_mats=1 )
#            for idx in range(maxIters-1) :
#                (junk_fx, junk_fz, junk_gx, g_z) = self.jac_system( mats )

#                Delta_z = -sparseSolve( sparseMat(g_z), g )
#                z_cons[:,idx+1] = z_cons[:,idx] + Delta_z

#                relres_z = numpy.linalg.norm(Delta_z,numpy.inf) / numpy.linalg.norm(z,numpy.inf)
#                if relres_z < tol :
#                    break
#                elif idx == maxIters-1 :
#                    print(('Warning: Max Newton iterations reached for consistency | RelChange=',relres_z*100.0))

#            z = z_cons[:,idx+1]

#        #print Cur

#        f, g = self.dae_system( deepcopy(x), deepcopy(z), Cur )

#        x_nxt[:,0] = deepcopy(x)
#        z_nxt[:,0] = deepcopy(z)
#        
#       # plt.figure(1)
#       # plt.plot( x_nxt[:,0] )
#       # plt.plot( z_nxt[:,0] )
#       # plt.show()

#        for idx in range(maxIters-1) :
#            f_nxt, g_nxt, mats = self.dae_system( x_nxt[:,idx], z_nxt[:,idx], Cur_nxt, get_mats=1  )

##            print 'x:',x.shape
##            print 'xnxt:',x_nxt[:,idx].shape
##            print 'f:',f.shape
##            print 'fnxt:',f_nxt.shape

##            print 'z:', z.shape
##            print 'g:', g.shape
##            print 'znxt:', z_nxt[:,idx].shape
##            print 'gnxt:', g_nxt.shape

#            F1 = x - x_nxt[:,idx] + delta_t/2.*( f+f_nxt )
#            F2 = g_nxt
#            F  = numpy.concatenate( (F1, F2), axis=0 )

#            fx, fz, gx, gz = self.jac_system( mats )


#            jmat = numpy.concatenate( (numpy.concatenate( (fx, fz), axis=1 ), 
#                                       numpy.concatenate( (gx, gz), axis=1 )) )

#            self.Input = Cur_nxt
#            jmat_num = compute_deriv( self.dae_system_num, numpy.concatenate( (x_nxt[:,idx], z_nxt[:,idx]) ) )

#            fx_num = jmat_num[:self.num_diff_vars,:self.num_diff_vars]
#            fz_num = jmat_num[:self.num_diff_vars,self.num_diff_vars:]
#            gx_num = jmat_num[self.num_diff_vars:,:self.num_diff_vars]
#            gz_num = jmat_num[self.num_diff_vars:,self.num_diff_vars:]

#            F1x_num = -sparse.eye(len(x)) + delta_t/2. * fx_num
#            F1z_num = delta_t/2. * fz_num

#            F1_x = -sparse.eye(len(x)) + delta_t/2. * fx
#            F1_z = delta_t/2. * fz
#            F2_x = gx
#            F2_z = gz

#            J = numpy.concatenate( (numpy.concatenate( (F1_x, F1_z), axis=1 ), 
#                                    numpy.concatenate( (F2_x, F2_z), axis=1 )) )

##            Jnum = numpy.concatenate( (numpy.concatenate( (F1x_num, F1z_num), axis=1 ), 
##                                       numpy.concatenate( (gx_num , gz_num ), axis=1 )) )


#            Jsp = sparseMat( J )

##            Jspnum = sparseMat( Jnum )

##            Delta_y = -sparseSolve( Jspnum, F )
#            Delta_y = -sparseSolve( Jsp, F )


#            x_nxt[:,idx+1] = x_nxt[:,idx] + Delta_y[:Nx]
#            z_nxt[:,idx+1] = z_nxt[:,idx] + Delta_y[Nx:]

#         #   plt.figure(1)
#          #  plt.plot(Delta_y)

#           # plt.figure(2)
#         #   plt.plot(x_nxt[:,idx])
#          #  plt.plot(x_nxt[:,idx+1])
#            
##            plt.show()

#            y = numpy.concatenate( (x_nxt[:,idx+1], z_nxt[:,idx+1]), axis=0 )
#            relres[idx+1] = numpy.linalg.norm( Delta_y, numpy.inf ) / numpy.linalg.norm( y, numpy.inf ) 

#            if (relres[idx+1]<tol) and (numpy.linalg.norm(F, numpy.inf)<tol) :
#                break
#            elif idx==maxIters-1 :
#                print( ('Warning: Max Newton iterations reached in main CN loop | RelChange = ',relres[-1]*100.0) )

#        x_nxtf = x_nxt[:,idx+1]
#        z_nxtf = z_nxt[:,idx+1]

#        newtonStats = {'var_flag':var_flag}
#        newtonStats['iters']    = idx
#        newtonStats['relres']   = relres

#        print '###############################################'
#        print 'numpy.allclose( fx, fx_num, rtol=0.001 ):', numpy.allclose( fx, fx_num, rtol=0.001 )
#        
#        print '###############################################'
#        print 'numpy.allclose( fz, fz_num, rtol=0.001 ):', numpy.allclose( fz, fz_num, rtol=0.001 )

#        print '###############################################'
#        print 'numpy.allclose( gx, gx_num, rtol=0.001 ):', numpy.allclose( gx, gx_num, rtol=0.001 )
#        
#        print '###############################################'
#        print 'numpy.allclose( gz, gz_num, rtol=0.001 ):', numpy.allclose( gz, gz_num, rtol=0.001 )

#        print '###############################################'
#        print 'numpy.allclose( jmat, jmat_num, rtol=0.001 ):', numpy.allclose( jmat, jmat_num, rtol=0.001 )

#        jm1_sp = sps.csr_matrix(jmat)
#        jm2_sp = sps.csr_matrix(jmat_num)

#        fig, ax = plt.subplots(1,2)
#        ax[0].spy( jm1_sp )
#        ax[0].set_title('Analytical Jacobian')
#        ax[1].spy( jm2_sp )
#        ax[1].set_title('Numerical Jacobian')
#        plt.suptitle( 'numpy.allclose( jmat, jmat_num, rtol=0.001 ):' + str(numpy.allclose( jmat, jmat_num, rtol=0.001 )) )
#        plt.show()

#        print 'Finished t_step'

#        return x_nxtf, z_nxtf, newtonStats
