import numpy
import scipy.linalg
import math

from assimulo.solvers import IDA
from assimulo.problem import Implicit_Problem

from copy import deepcopy

# battsimpy modules
import params
from helper_modules import batteqns


class Full_1D(Implicit_Problem):
    """
    Class for the distributed full Pseudo-2D Newman-style model.
    The distributed variant of the P2D model provides the ability to connect
    multiple instances of the model electrically in parallel, and run a
    simulation. This was developed to simulate the conditions that occur for
    multiple cells connected in parallel when there is some reason to have
    different impedances among the cells. For example, non-uniform temperature
    distributions among the cells could be the cause of this.

    Important methods:
        res()
            Contains the residual for the model.
        jac()
            Contains the analytical Jacobian for the model.
        These two methods must be named "res" and "jac" to be noticed properly
        by Assimulo.
    """

    def __init__(self, p, y0, yd0):
        self.p = p
        self.pars = self.p

        self.Ac = p.Ac

        self.T_amb = 30. + 273.15
        self.T = 30. + 273.15  # Cell temperature, [K]

        self.phie_mats()
        self.phis_mats()
        self.cs_mats()

        # System indices
        self.ce_inds = range(p.N)
        self.ce_inds_r = numpy.reshape(self.ce_inds, [len(self.ce_inds), 1])
        self.ce_inds_c = numpy.reshape(self.ce_inds, [1, len(self.ce_inds)])

        self.csa_inds = range(p.N, p.N + (p.Na * p.Nra))
        self.csa_inds_r = numpy.reshape(self.csa_inds, [len(self.csa_inds), 1])
        self.csa_inds_c = numpy.reshape(self.csa_inds, [1, len(self.csa_inds)])

        self.csc_inds = range(p.N + (p.Na * p.Nra),
                              p.N + (p.Na * p.Nra) + (p.Nc * p.Nrc))
        self.csc_inds_r = numpy.reshape(self.csc_inds, [len(self.csc_inds), 1])
        self.csc_inds_c = numpy.reshape(self.csc_inds, [1, len(self.csc_inds)])

        self.T_ind = p.N + (p.Na * p.Nra) + (p.Nc * p.Nrc)

        c_end = p.N + (p.Na * p.Nra) + (p.Nc * p.Nrc) + 1

        self.ja_inds = range(c_end, c_end + p.Na)
        self.ja_inds_r = numpy.reshape(self.ja_inds, [len(self.ja_inds), 1])
        self.ja_inds_c = numpy.reshape(self.ja_inds, [1, len(self.ja_inds)])

        self.jc_inds = range(c_end + p.Na, c_end + p.Na + p.Nc)
        self.jc_inds_r = numpy.reshape(self.jc_inds, [len(self.jc_inds), 1])
        self.jc_inds_c = numpy.reshape(self.jc_inds, [1, len(self.jc_inds)])

        self.pe_inds = range(c_end + p.Na + p.Nc, c_end + p.Na + p.Nc + p.N)
        self.pe_inds_r = numpy.reshape(self.pe_inds, [len(self.pe_inds), 1])
        self.pe_inds_c = numpy.reshape(self.pe_inds, [1, len(self.pe_inds)])

        self.pe_a_inds = range(c_end + p.Na + p.Nc, c_end + p.Na + p.Nc + p.Na)
        self.pe_a_inds_r = numpy.reshape(
            self.pe_a_inds, [len(self.pe_a_inds), 1])
        self.pe_a_inds_c = numpy.reshape(
            self.pe_a_inds, [1, len(self.pe_a_inds)])

        self.pe_c_inds = range(
            c_end + p.Na + p.Nc + p.Na + p.Ns,
            c_end + p.Na + p.Nc + p.N)
        self.pe_c_inds_r = numpy.reshape(
            self.pe_c_inds, [len(self.pe_c_inds), 1])
        self.pe_c_inds_c = numpy.reshape(
            self.pe_c_inds, [1, len(self.pe_c_inds)])

        self.pa_inds = range(
            c_end + p.Na + p.Nc + p.N,
            c_end + p.Na + p.Nc + p.N + p.Na)
        self.pa_inds_r = numpy.reshape(self.pa_inds, [len(self.pa_inds), 1])
        self.pa_inds_c = numpy.reshape(self.pa_inds, [1, len(self.pa_inds)])

        self.pc_inds = range(
            c_end + p.Na + p.Nc + p.N + p.Na,
            c_end + p.Na + p.Nc + p.N + p.Na + p.Nc)
        self.pc_inds_r = numpy.reshape(self.pc_inds, [len(self.pc_inds), 1])
        self.pc_inds_c = numpy.reshape(self.pc_inds, [1, len(self.pc_inds)])

        # second set for manual jac version
        c_end = 0
        self.ja_inds2 = range(c_end, c_end + p.Na)
        self.ja_inds_r2 = numpy.reshape(self.ja_inds2, [len(self.ja_inds2), 1])
        self.ja_inds_c2 = numpy.reshape(self.ja_inds2, [1, len(self.ja_inds2)])

        self.jc_inds2 = range(c_end + p.Na, c_end + p.Na + p.Nc)
        self.jc_inds_r2 = numpy.reshape(self.jc_inds2, [len(self.jc_inds2), 1])
        self.jc_inds_c2 = numpy.reshape(self.jc_inds2, [1, len(self.jc_inds2)])

        self.pe_inds2 = range(c_end + p.Na + p.Nc, c_end + p.Na + p.Nc + p.N)
        self.pe_inds_r2 = numpy.reshape(self.pe_inds2, [len(self.pe_inds2), 1])
        self.pe_inds_c2 = numpy.reshape(self.pe_inds2, [1, len(self.pe_inds2)])

        self.pe_a_inds2 = range(
            c_end + p.Na + p.Nc,
            c_end + p.Na + p.Nc + p.Na)
        self.pe_a_inds_r2 = numpy.reshape(
            self.pe_a_inds2, [len(self.pe_a_inds2), 1])
        self.pe_a_inds_c2 = numpy.reshape(
            self.pe_a_inds2, [1, len(self.pe_a_inds2)])

        self.pe_c_inds2 = range(
            c_end + p.Na + p.Nc + p.Na + p.Ns,
            c_end + p.Na + p.Nc + p.N)
        self.pe_c_inds_r2 = numpy.reshape(
            self.pe_c_inds2, [len(self.pe_c_inds2), 1])
        self.pe_c_inds_c2 = numpy.reshape(
            self.pe_c_inds2, [1, len(self.pe_c_inds2)])

        self.pa_inds2 = range(
            c_end + p.Na + p.Nc + p.N,
            c_end + p.Na + p.Nc + p.N + p.Na)
        self.pa_inds_r2 = numpy.reshape(self.pa_inds2, [len(self.pa_inds2), 1])
        self.pa_inds_c2 = numpy.reshape(self.pa_inds2, [1, len(self.pa_inds2)])

        self.pc_inds2 = range(
            c_end + p.Na + p.Nc + p.N + p.Na,
            c_end + p.Na + p.Nc + p.N + p.Na + p.Nc)
        self.pc_inds_r2 = numpy.reshape(self.pc_inds2, [len(self.pc_inds2), 1])
        self.pc_inds_c2 = numpy.reshape(self.pc_inds2, [1, len(self.pc_inds2)])

        # Matrices for thermal calcs (gradient operators)
        self.Ga, self.Gc, self.G = batteqns.grad_mat(
            p.Na, p.x_m_a), batteqns.grad_mat(
            p.Nc, p.x_m_c), batteqns.grad_mat(
            p.N, p.x_m)

        # Initialize the C arrays for the heat generation (these are useful for
        # the Jacobian)
        junkQ = self.calc_heat(y0,
                               numpy.zeros(p.Na),
                               numpy.zeros(p.Nc),
                               p.uref_a(y0[self.csa_inds[:p.Na]] / p.csa_max),
                               p.uref_c(y0[self.csc_inds[:p.Nc]] / p.csc_max))

        # Kinetic C array (also useful for the Jacobian)
        csa_ss = y0[self.csa_inds[:p.Na]]
        csc_ss = y0[self.csc_inds[:p.Nc]]
        ce = y0[self.ce_inds]
        T = y0[self.T_ind]

        self.ioa_const = self.pars.ioa_const
        self.ioc_const = self.pars.ioc_const

        self.C_ioa = (2.0 * self.ioa_const
                      * math.exp(self.pars.ioa_Ea
                                 / 8.314
                                 * (1. / 298.15 - 1. / T))
                      / p.F
                      * numpy.sqrt(ce[:p.Na] / p.ce_nom))
        # * (1.0 - csa_ss/p.csa_max) * (csa_ss/p.csa_max) )

        self.C_ioc = (2.0 * self.ioc_const
                      * math.exp(self.pars.ioc_Ea
                                 / 8.314
                                 * (1. / 298.15 - 1. / T))
                      / p.F
                      * numpy.sqrt(ce[-p.Nc:] / p.ce_nom
                                   * (1.0 - csc_ss / p.csc_max)
                                   * (csc_ss / p.csc_max)))

#        self.C_ioa = (2.0*self.io_a/self.F) * numpy.ones_like( csa_ss )
#        self.C_ioc = (2.0*self.io_a/self.F) * numpy.ones_like( csc_ss )

    def setup_model(self, y0, yd0, name):
        """
        Setup the Assimulo implicit model to use IDA.
        """
        Implicit_Problem.__init__(self, y0=y0, yd0=yd0, name=name)

    def phie_mats(self,):
        """
        Electrolyte constant B_ce matrix
        """
        p = self.p

        Ba = [
            (1. - p.t_plus) * asa / ea for ea,
            asa in zip(
                p.eps_a_vec,
                p.as_a)]
        Bs = [0.0 for i in range(p.Ns)]
        Bc = [
            (1. - p.t_plus) * asc / ec for ec,
            asc in zip(
                p.eps_c_vec,
                p.as_c)]

        self.B_ce = numpy.diag(numpy.array(Ba + Bs + Bc, dtype='d'))

        Bap = [asa * p.F for asa in p.as_a]
        Bsp = [0.0 for i in range(p.Ns)]
        Bcp = [asc * p.F for asc in p.as_c]

        self.B2_pe = numpy.diag(numpy.array(Bap + Bsp + Bcp, dtype='d'))

    def phis_mats(self,):
        """
        Solid phase parameters and j vector matrices
        """
        p = self.p

        self.A_ps_a = batteqns.flux_mat_builder(
            p.Na, p.x_m_a, numpy.ones_like(
                p.vols_a), p.sig_a_eff)
        self.A_ps_c = batteqns.flux_mat_builder(
            p.Nc, p.x_m_c, numpy.ones_like(
                p.vols_c), p.sig_c_eff)

        Baps = numpy.array(
            [asa * p.F * dxa for asa, dxa in zip(p.as_a, p.vols_a)], dtype='d')
        Bcps = numpy.array(
            [asc * p.F * dxc for asc, dxc in zip(p.as_c, p.vols_c)], dtype='d')

        self.B_ps_a = numpy.diag(Baps)
        self.B_ps_c = numpy.diag(Bcps)

        self.B2_ps_a = numpy.zeros(p.Na, dtype='d')
        self.B2_ps_a[0] = -1.
        self.B2_ps_c = numpy.zeros(p.Nc, dtype='d')
        self.B2_ps_c[-1] = -1.

    def cs_mats(self,):
        """
        Intiliaze the solid phase diffusion model matrices.
        """
        p = self.p

        # 1D spherical diffusion model
        # A_cs pre build
        self.A_csa_single = batteqns.flux_mat_builder(
            p.Nra, p.r_m_a, p.vols_ra_m, p.Dsa * (p.r_e_a**2))
        self.A_csc_single = batteqns.flux_mat_builder(
            p.Nrc, p.r_m_c, p.vols_rc_m, p.Dsc * (p.r_e_c**2))

        # A_cs build up to the stacked full cs size (Nr and Nx)
        b = [self.A_csa_single] * p.Na
        self.A_cs_a = scipy.linalg.block_diag(*b)
        b = [self.A_csc_single] * p.Nc
        self.A_cs_c = scipy.linalg.block_diag(*b)

        # B_cs and C_cs are constant (i.e., are not state-dependent)
        self.B_csa_single = numpy.array([0. for i in range(
            p.Nra - 1)] + [-1. * p.r_e_a[-1]**2 / p.vols_ra_m[-1]], dtype='d')
        self.B_csc_single = numpy.array([0. for i in range(
            p.Nrc - 1)] + [-1. * p.r_e_c[-1]**2 / p.vols_rc_m[-1]], dtype='d')

        b = [self.B_csa_single] * p.Na
        self.B_cs_a = scipy.linalg.block_diag(*b).T
        b = [self.B_csc_single] * p.Nc
        self.B_cs_c = scipy.linalg.block_diag(*b).T

        # Particle surface concentration
        h_na = p.r_e_a[-1] - p.r_m_a[-1]
        h_n1a = p.r_m_a[-1] - p.r_m_a[-2]

        h_nc = p.r_e_c[-1] - p.r_m_c[-1]
        h_n1c = p.r_m_c[-1] - p.r_m_c[-2]

        self.a_n_a, self.b_n_a, self.c_n_a = batteqns.right_side_coeffs(
            h_na, h_n1a)
        self.a_n_c, self.b_n_c, self.c_n_c = batteqns.right_side_coeffs(
            h_nc, h_n1c)

        self.C_cs_a_single = numpy.array(
            [0. for i in range(p.Nra - 2)]
            + [-self.a_n_a / self.c_n_a, -self.b_n_a / self.c_n_a],
            dtype='d')
        self.C_cs_c_single = numpy.array(
            [0. for i in range(p.Nrc - 2)]
            + [-self.a_n_c / self.c_n_c, -self.b_n_c / self.c_n_c],
            dtype='d')

        self.C_cs_a = scipy.linalg.block_diag(*[self.C_cs_a_single] * p.Na)
        self.C_cs_c = scipy.linalg.block_diag(*[self.C_cs_c_single] * p.Nc)

        self.C_cs_a_avg = scipy.linalg.block_diag(
            *[1. / ((1. / 3.) * p.Rp_a**3) * p.vols_ra_m] * p.Na)
        self.C_cs_c_avg = scipy.linalg.block_diag(
            *[1. / ((1. / 3.) * p.Rp_c**3) * p.vols_rc_m] * p.Nc)

        self.C_cs_a_mean = 1. / p.La * p.vols_a.dot(self.C_cs_a_avg)
        self.C_cs_c_mean = 1. / p.Lc * p.vols_c.dot(self.C_cs_c_avg)

        # Particle core concentration
        h_na = p.r_e_a[0] - p.r_m_a[0]
        h_n1a = p.r_m_a[1] - p.r_m_a[0]

        h_nc = p.r_e_c[0] - p.r_m_c[0]
        h_n1c = p.r_m_c[1] - p.r_m_c[0]

        a_n_a, b_n_a, c_n_a = batteqns.left_side_coeffs(h_na, h_n1a)
        a_n_c, b_n_c, c_n_c = batteqns.left_side_coeffs(h_nc, h_n1c)

        C_cso_a_single = numpy.array(
            [-b_n_a / a_n_a, -c_n_a / a_n_a]
            + [0. for i in range(p.Nra - 2)], dtype='d')
        C_cso_c_single = numpy.array(
            [-b_n_c / a_n_c, -c_n_c / a_n_c]
            + [0. for i in range(p.Nrc - 2)], dtype='d')

        self.C_cso_a = scipy.linalg.block_diag(*[C_cso_a_single] * p.Na)
        self.C_cso_c = scipy.linalg.block_diag(*[C_cso_c_single] * p.Nc)

        # D_cs prelim values, note this is Ds(cs) dependent and therefore
        # requires updating for state dependent Ds
        self.D_cs_a = -1.0 / (p.Dsa * self.c_n_a) * numpy.eye(p.Na)
        self.D_cs_c = -1.0 / (p.Dsc * self.c_n_c) * numpy.eye(p.Nc)

    def set_iapp(self, i_app):
        """
        Calculate and assign the applied input current density.
        """
        self.i_app = i_app

    def update_cs_mats(self, csa, csc, csa_ss, csc_ss, csa_o, csc_o, T):
        """
        FVM discretization of the concentration flux term for the solid
        particle diffusion equation.
        A matrix is generated for each particle that exists at each node point
        along the mesh in the thickness of each electrode.

        In general, the diffusivity is a function of concentration, and
        therefore these matrices are different for each particle, and change
        through time.
        If, Ds(cs) is constant, then this only needs to be initialized and
        then may be left constat during the simulation.
        """
        p = self.p

        Acsa_list = [[] for i in range(p.Na)]
        Acsc_list = [[] for i in range(p.Nc)]

        Dsa_ss = [0. for i in range(p.Na)]
        Dsc_ss = [0. for i in range(p.Nc)]

        for ia in range(p.Na):

            csa_m = csa[ia * p.Nra:(ia + 1) * p.Nra]
            csa_e = numpy.array(
                [csa_o[ia]] + [0.5 * (csa_m[i + 1] + csa_m[i])
                               for i in range(p.Nra - 1)] + [csa_ss[ia]])

            Ua_e = p.uref_a(csa_e / p.csa_max)
            Dsa_e = p.Dsa_intp(Ua_e) \
                * math.exp(p.Ea_Dsa / p.R_gas * (1. / 298.15 - 1. / T))

            Acsa_list[ia] = batteqns.flux_mat_builder(p.Nra, p.r_m_a,
                                                      p.vols_ra_m,
                                                      Dsa_e * (p.r_e_a**2))

            Dsa_ss[ia] = Dsa_e[-1]

        for ic in range(p.Nc):
            csc_m = csc[ic * p.Nrc:(ic + 1) * p.Nrc]
            csc_e = numpy.array(
                [csc_o[ic]] + [0.5 * (csc_m[i + 1] + csc_m[i])
                               for i in range(p.Nrc - 1)] + [csc_ss[ic]])

            Uc_e = p.uref_c(csc_e / p.csc_max)
            Dsc_e = p.Dsc_intp(Uc_e) * math.exp(p.Ea_Dsc
                                                / p.R_gas
                                                * (1. / 298.15 - 1. / T))

            Acsc_list[ic] = batteqns.flux_mat_builder(p.Nrc, p.r_m_c,
                                                      p.vols_rc_m,
                                                      Dsc_e * (p.r_e_c**2))

            Dsc_ss[ic] = Dsc_e[-1]

        self.A_cs_a = scipy.linalg.block_diag(*Acsa_list)
        self.A_cs_c = scipy.linalg.block_diag(*Acsc_list)

        self.D_cs_a = numpy.diag(-1.0 / (numpy.array(Dsa_ss) * self.c_n_a))
        self.D_cs_c = numpy.diag(-1.0 / (numpy.array(Dsc_ss) * self.c_n_c))

    # Define c_e functions
    def build_Ace_mat(self, c, T):
        """
        FVM discretization of the concentration flux term for the elyte
        diffusion equation.
        """
        p = self.p

        D_eff = self.Diff_ce(c, T)

        A = p.K_m.dot(batteqns.flux_mat_builder(p.N, p.x_m, p.vols, D_eff))

        return A

    def Diff_ce(self, c, T, mid_on=0, eps_off=0):
        """
        Elyte diffusivity interpolator.
        Function of c_e and T.
        """
        p = self.p

        D_ce = p.De_intp(c, T, grid=False).flatten()

        if eps_off:
            D_mid = D_ce
        else:
            D_mid = D_ce * p.eps_eff

        if mid_on:
            D_out = D_mid
        else:
            if isinstance(c, float):
                D_out = D_mid
            else:
                D_out = batteqns.mid_to_edge(D_mid, p.x_e)

        return D_out

    # Define phi_e functions
    def build_Ape_mat(self, c, T):
        """
        FVM discretization of the potential flux term for the elyte potential
        equation.
        """
        p = self.p

        k_eff = self.kapp_ce(c, T)

        A = batteqns.flux_mat_builder(p.N, p.x_m, p.vols, k_eff)

        A[-1, -1] = 2 * A[-1, -1]  # BC update for phi_e = 0

        return A

    def build_Bpe_mat(self, c, T):
        """
        FVM discretization of the concentration flux term for the elyte
        potential equation.
        """
        p = self.p

        gam = 2. * (1. - p.t_plus) * p.R_gas * T / p.F

        k_eff = self.kapp_ce(c, T)

        c_edge = batteqns.mid_to_edge(c, p.x_e)

        B1 = batteqns.flux_mat_builder(
            p.N, p.x_m, p.vols, k_eff * gam / c_edge)

        return B1

    def kapp_ce(self, c, T, mid_on=0, eps_off=0):
        """
        Elyte conductivity interpolator.
        Function of c_e and T.
        """
        p = self.p

        # 1e-1 converts from mS/cm to S/m (model uses SI units)
        k_ce = 1e-1 * p.ke_intp(c, T, grid=False).flatten()

        if eps_off:
            k_mid = k_ce
        else:
            k_mid = k_ce * p.eps_eff

        if mid_on:
            k_out = k_mid
        else:
            if isinstance(c, float):
                k_out = k_mid
            else:
                k_out = batteqns.mid_to_edge(k_mid, p.x_e)

        return k_out

    def build_Bjac_mat(self, eta, a, b):
        """
        Jacobian matrix for the Butler-Volmer kinetics equation. d/dc
        """
        d = a * numpy.cosh(b * eta) * b

        return numpy.diag(d)

    def build_BjT_mat(self, T, a, b):
        """
        Jacobian matrix for the Butler-Volmer kinetics equation. d/dT
        """
        d = a * numpy.cosh(b / T) * (-b / T**2)

        return d

    def get_voltage(self, y):
        """
        Return the cell potential at the terminals

        An additional ohmic loss is included by the foil resistance
        (current collectors), Rfl, and the tab resistance, Rtb.
        These parameters are set in the model config file.
        """
        pc = y[self.pc_inds]
        pa = y[self.pa_inds]

        Vcell = pc[-1] - pa[0]

        return Vcell - (self.pars.Rfl + self.pars.Rtb) * \
            (self.pars.Ac * self.i_app)

    def get_eta_uref(self, csa, csc, ja_rxn, jc_rxn, phi_s_a, phi_s_c, phi):
        """
        Calcuate the reaction kinetics overpotential on the anode and cathode.
        Provide the overpotentials, and other variables calc'd in the process.
        """
        p = self.p

        csa_ss = (self.C_cs_a.dot(csa)).flatten() + \
            (self.D_cs_a.dot(ja_rxn)).flatten()
        csc_ss = (self.C_cs_c.dot(csc)).flatten() + \
            (self.D_cs_c.dot(jc_rxn)).flatten()

        Uref_a = p.uref_a(csa_ss / p.csa_max)  # anode   equilibrium potential
        Uref_c = p.uref_c(csc_ss / p.csc_max)  # cathode equilibrium potential

        eta_a = phi_s_a - phi[:p.Na] - Uref_a  # anode   overpotential
        eta_c = phi_s_c - phi[-p.Nc:] - Uref_c  # cathode overpotential

        return eta_a, eta_c, Uref_a, Uref_c, csa_ss, csc_ss

    def update_Cio(self, csa_ss, csc_ss, ce, T):
        """
        Update the matrices used for the exchange current densities in the
        Jacobian.
        In general, io is a function of solid conc, elyte conc, and
        temperature.
        """
        p = self.p

        self.C_ioa = (2.0 * self.ioa_const
                      * math.exp(self.pars.ioa_Ea
                                 / 8.314
                                 * (1. / 298.15 - 1. / T))
                      / p.F * numpy.sqrt(ce[:p.Na] / p.ce_nom))
        # * (1.0 - csa_ss/p.csa_max) * (csa_ss/p.csa_max) )
        self.C_ioc = (2.0 * self.ioc_const
                      * math.exp(self.pars.ioc_Ea
                                 / 8.314
                                 * (1. / 298.15 - 1. / T))
                      / p.F * numpy.sqrt(ce[-p.Nc:] / p.ce_nom
                                         * (1.0 - csc_ss / p.csc_max)
                                         * (csc_ss / p.csc_max)))

    def calc_heat(self, y, eta_a, eta_c, Uref_a, Uref_c):
        """
        y       :state vector
        eta_a,c :anode/cathode kinetic overpotential
        Uref_a,c:anode/cathode equilibrium potential at surface of each
        particle through thickness of mesh.

        Return the total, spatially integrated, heat source across the cell
        sandwich.
        """
        p = self.p

        ce = y[self.ce_inds]
        csa = y[self.csa_inds]
        csc = y[self.csc_inds]
        ja = y[self.ja_inds]
        jc = y[self.jc_inds]
        phi = y[self.pe_inds]
        phi_s_a = y[self.pa_inds]
        phi_s_c = y[self.pc_inds]
        T = y[self.T_ind]

        # Gradients for heat calc
        dphi_s_a = numpy.gradient(phi_s_a) / numpy.gradient(p.x_m_a)
        dphi_s_c = numpy.gradient(phi_s_c) / numpy.gradient(p.x_m_c)

        dphi = numpy.gradient(phi) / numpy.gradient(p.x_m)

        dlnce = 1. / ce * (numpy.gradient(ce) / numpy.gradient(p.x_m))

        # kapp_eff at the node points (middle of control volume, rather than
        # edge)
        kapp_eff_m = self.kapp_ce(ce, T, mid_on=1)

        # Reaction kinetics heat
        C_ra = (p.vols_a * p.F * p.as_a)
        C_rc = (p.vols_c * p.F * p.as_c)

        Q_rxn_a = C_ra.dot(ja * eta_a)
        Q_rxn_c = C_rc.dot(jc * eta_c)
        Q_rxn = Q_rxn_a + Q_rxn_c

        csa_mean = self.C_cs_a_avg.dot(csa)
        csc_mean = self.C_cs_c_avg.dot(csc)
        Uam = p.uref_a(csa_mean / p.csa_max)
        Ucm = p.uref_c(csc_mean / p.csc_max)

        eta_conc_a = Uref_a - Uam
        eta_conc_c = Uref_c - Ucm

        Q_conc_a = C_ra.dot(eta_conc_a * ja)
        Q_conc_c = C_rc.dot(eta_conc_c * jc)

        Q_conc = Q_conc_a + Q_conc_c

        # Ohmic heat in electrolyte and solid
        C_pe = (
            p.vols.dot(
                numpy.diag(
                    kapp_eff_m *
                    dphi).dot(
                    self.G)) +
            p.vols.dot(
                numpy.diag(
                    2 *
                    kapp_eff_m *
                    p.R_gas *
                    T /
                    p.F *
                    (
                        1. -
                        p.t_plus) *
                    dlnce).dot(
                    self.G)))

        Q_ohm_e = C_pe.dot(phi)

        C_pa = p.vols_a.dot(numpy.diag(p.sig_a_eff * dphi_s_a).dot(self.Ga))
        C_pc = p.vols_c.dot(numpy.diag(p.sig_c_eff * dphi_s_c).dot(self.Gc))

        Q_ohm_s = C_pa.dot(phi_s_a) + C_pc.dot(phi_s_c)

        Q_ohm = Q_ohm_e + Q_ohm_s

        # Entropic heat
        # ??

        # Total heat
        Q_tot = Q_ohm + Q_rxn + Q_conc

        self.C_q_pe = C_pe
        self.C_q_pa = C_pa
        self.C_q_pc = C_pc
        self.C_q_na = C_ra * ja
        self.C_q_nc = C_rc * jc
        self.C_q_ja = C_ra * eta_a + C_ra * eta_conc_a
        self.C_q_jc = C_rc * eta_c + C_rc * eta_conc_c

        return Q_tot

    def res(self, t, y, yd):
        """
        Residual for the full 1d model.
        """
        p = self.p

        # Parse out the states
        # E-lyte conc
        ce = y[self.ce_inds]
        c_dots = yd[self.ce_inds]

        # Solid conc a:anode, c:cathode
        csa = y[self.csa_inds]
        csc = y[self.csc_inds]
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
        T = y[self.T_ind]
        T_dt = yd[self.T_ind]

        # Grab state dependent matrices
        # For E-lyte conc and potential (i.e., De(ce), kapp_e(ce))
        A_ce = self.build_Ace_mat(ce, T)
        A_pe = self.build_Ape_mat(ce, T)
        B_pe = self.build_Bpe_mat(ce, T)

        # Compute extra variables
        # For the reaction kinetics
        eta_a, eta_c, Uref_a, Uref_c, csa_ss, csc_ss = self.get_eta_uref(
            csa, csc, ja_rxn, jc_rxn, phi_s_a, phi_s_c, phi)

        # For Solid conc Ds
        csa_o = (self.C_cso_a.dot(csa)).flatten()
        csc_o = (self.C_cso_c.dot(csc)).flatten()

        self.update_cs_mats(csa, csc, csa_ss, csc_ss, csa_o, csc_o, T)

        # For kinetics, the io param is now conc dependent
        self.update_Cio(csa_ss, csc_ss, ce, T)

        Q_in = self.calc_heat(y, eta_a, eta_c, Uref_a, Uref_c)

        Q_out = p.h * p.Aconv * (T - self.T_amb)

        ja = self.C_ioa * numpy.sinh(0.5 * p.F / (p.R_gas * T) * eta_a)
        jc = self.C_ioc * numpy.sinh(0.5 * p.F / (p.R_gas * T) * eta_c)

        j = numpy.concatenate([ja_rxn, numpy.zeros(p.Ns), jc_rxn])

        # Compute the residuals
        # Time deriv components
        r1 = c_dots - (((A_ce.dot(ce)).flatten()
                        + (self.B_ce.dot(j)).flatten()))  # E-lyte conc

        r2 = csa_dt - (self.A_cs_a.dot(csa).flatten()
                       + self.B_cs_a.dot(ja_rxn).flatten())  # Anode   conc
        r3 = csc_dt - (self.A_cs_c.dot(csc).flatten()
                       + self.B_cs_c.dot(jc_rxn).flatten())  # Cathode conc

        r4 = T_dt - 1. / (p.rho * p.Cp) * (Q_in - Q_out)

        # Algebraic components -- Butler-Volmer reaction kinetics
        r5 = ja_rxn - ja
        r6 = jc_rxn - jc

        # Algebraic components -- E-lyte potential
        r7 = A_pe.dot(phi).flatten() - B_pe.dot(ce).flatten() + \
            self.B2_pe.dot(j).flatten()

        # Algebraic components -- Anode potential
        r8 = self.A_ps_a.dot(phi_s_a).flatten() \
            - self.B_ps_a.dot(ja_rxn).flatten() \
            - self.B2_ps_a * self.i_app
        # Algebraic components -- Cathode potential
        r9 = self.A_ps_c.dot(phi_s_c).flatten() \
            - self.B_ps_c.dot(jc_rxn).flatten() \
            + self.B2_ps_c * self.i_app

        res_out = numpy.concatenate([r1, r2, r3, [r4], r5, r6, r7, r8, r9])

        return res_out

    def jac(self, c, t, y, yd):
        """
        Analytical Jacobian for the full 1d model residual.
        """
        p = self.p

        # Setup
        # Parse out the states
        # E-lyte conc
        ce = y[self.ce_inds]

        # Solid conc a:anode, c:cathode
        csa = y[self.csa_inds]
        csc = y[self.csc_inds]

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

        # Grab state dependent matrices
        # For E-lyte conc and potential (i.e., De(ce), kapp_e(ce))
        A_ce = self.build_Ace_mat(ce, T)
        A_pe = self.build_Ape_mat(ce, T)
        B_pe = self.build_Bpe_mat(ce, T)

        # Compute extra variables
        # For the reaction kinetics
        eta_a, eta_c, Uref_a, Uref_c, csa_ss, csc_ss = self.get_eta_uref(
            csa, csc, ja_rxn, jc_rxn, phi_s_a, phi_s_c, phi)

        # Build the Jac matrix
        # Self coupling
        A_dots = numpy.diag([1 * c for i in range(p.num_diff_vars)])
        j_c = A_dots - scipy.linalg.block_diag(A_ce,
                                               self.A_cs_a,
                                               self.A_cs_c,
                                               [-p.h * p.Aconv / p.rho / p.Cp])

        Bjac_a = self.build_Bjac_mat(
            eta_a, self.C_ioa, 0.5 * p.F / (p.R_gas * T))
        Bjac_c = self.build_Bjac_mat(
            eta_c, self.C_ioc, 0.5 * p.F / (p.R_gas * T))

        BjT_a = self.build_BjT_mat(
            T, self.C_ioa, 0.5 * p.F / (p.R_gas) * eta_a)
        BjT_c = self.build_BjT_mat(
            T, self.C_ioc, 0.5 * p.F / (p.R_gas) * eta_c)

        dcss_dja = numpy.diagonal(self.D_cs_a)
        dcss_djc = numpy.diagonal(self.D_cs_c)

        dU_csa_ss = (1.0 / p.csa_max) * p.duref_a(csa_ss / p.csa_max)
        dU_csc_ss = (1.0 / p.csc_max) * p.duref_c(csc_ss / p.csc_max)

        DUDcsa_ss = numpy.diag(dU_csa_ss)
        DUDcsc_ss = numpy.diag(dU_csc_ss)

        A_ja = numpy.diag(numpy.ones(p.Na)) - \
            (Bjac_a.dot(-1.0 * DUDcsa_ss * 1.0)).dot(self.D_cs_a)
        A_jc = numpy.diag(numpy.ones(p.Nc)) - \
            (Bjac_c.dot(-1.0 * DUDcsc_ss * 1.0)).dot(self.D_cs_c)

        j = scipy.linalg.block_diag(
            j_c, A_ja, A_jc, A_pe, self.A_ps_a, self.A_ps_c)

        # Cross coupling
        # c_e: j coupling back in
        j[numpy.ix_(self.ce_inds, self.ja_inds)] = -self.B_ce[:, :p.Na]
        j[numpy.ix_(self.ce_inds, self.jc_inds)] = -self.B_ce[:, -p.Nc:]

        # cs_a: j coupling
        j[numpy.ix_(self.csa_inds, self.ja_inds)] = -self.B_cs_a
        # cs_c: j coupling
        j[numpy.ix_(self.csc_inds, self.jc_inds)] = -self.B_cs_c

        a_coeff = 2.0 * self.C_q_na * (-1.0) * dU_csa_ss
        Ca_T = numpy.array(
            [self.C_cs_a_single * ac for ac in a_coeff]).flatten()
        c_coeff = 2.0 * self.C_q_nc * (-1.0) * dU_csc_ss
        Cc_T = numpy.array(
            [self.C_cs_c_single * cc for cc in c_coeff]).flatten()

        # T
        j[self.T_ind, self.ja_inds] = (-1. / (p.rho * p.Cp)
                                       * (self.C_q_ja
                                          + 2.0 * (self.C_q_na * (-1.0)
                                                   * dU_csa_ss * dcss_dja)))
        j[self.T_ind, self.jc_inds] = (-1. / (p.rho * p.Cp)
                                       * (self.C_q_jc
                                          + 2.0 * (self.C_q_nc * (-1.0)
                                                   * dU_csc_ss * dcss_djc)))
        j[self.T_ind, self.pe_inds] \
            = (-1. / (p.rho * p.Cp)
               * (self.C_q_pe + numpy.array(list(self.C_q_na)
                                            + [0. for i in range(p.Ns)]
                                            + list(self.C_q_nc)) * (-1.0)))
        j[self.T_ind, self.pa_inds] = (-1. / (p.rho * p.Cp)
                                       * (self.C_q_pa + self.C_q_na * (1.0)))
        j[self.T_ind, self.pc_inds] = (-1. / (p.rho * p.Cp)
                                       * (self.C_q_pc + self.C_q_nc * (1.0)))
        j[self.T_ind, self.csa_inds] = -1. / (p.rho * p.Cp) * (Ca_T)
        j[self.T_ind, self.csc_inds] = -1. / (p.rho * p.Cp) * (Cc_T)

        j[self.ja_inds, self.T_ind] = -BjT_a
        j[self.jc_inds, self.T_ind] = -BjT_c

        # j_a: pe, pa, csa  coupling
        j[numpy.ix_(self.ja_inds, self.pa_inds)] = -Bjac_a * (1.0)
        j[numpy.ix_(self.ja_inds, self.pe_a_inds)] = -Bjac_a * (-1.0)
        j[numpy.ix_(self.ja_inds, self.csa_inds)]\
            = -(Bjac_a.dot(-1.0 * DUDcsa_ss * 1.0)).dot(self.C_cs_a)

        # j_c: pe, pc, csc  coupling
        j[numpy.ix_(self.jc_inds, self.pc_inds)] = -Bjac_c * (1.0)
        j[numpy.ix_(self.jc_inds, self.pe_c_inds)] = -Bjac_c * (-1.0)
        j[numpy.ix_(self.jc_inds, self.csc_inds)]\
            = -(Bjac_c.dot(-1.0 * DUDcsc_ss * 1.0)).dot(self.C_cs_c)

        # phi_e: ce coupling into phi_e equation
        j[numpy.ix_(self.pe_inds, self.ce_inds)] = -B_pe
        j[numpy.ix_(self.pe_inds, self.ja_inds)] = self.B2_pe[:, :p.Na]
        j[numpy.ix_(self.pe_inds, self.jc_inds)] = self.B2_pe[:, -p.Nc:]

        # phi_s_a: ja
        j[numpy.ix_(self.pa_inds, self.ja_inds)] = -self.B_ps_a
        # phi_s_c: jc
        j[numpy.ix_(self.pc_inds, self.jc_inds)] = -self.B_ps_c

        return j


class Results_object():
    """
    result properties at each schedule step
    """

    def __init__(self, p):
        """
        Define the basic properties for the results
        """
        self.c_s_a = []
        self.c_s_c = []
        self.c_e = []
        self.T = []
        self.phi_s_a = []
        self.phi_s_c = []
        self.phi_e = []
        self.ja = []
        self.jc = []

        self.csa_ss = []
        self.csc_ss = []
        self.csa_avg = []
        self.csc_avg = []

        self.Ua_avg = []
        self.Uc_avg = []
        self.Ua_ss = []
        self.Uc_ss = []
        self.eta_a = []
        self.eta_c = []

        self.Va = []
        self.Vc = []
        self.css_fullx = []
        self.Uss_fullx = []
        self.phis_fullx = []
        self.eta_fullx = []
        self.j_fullx = []

        self.ke = []
        self.De = []

        self.Volt = []
        self.Cur = []
        self.Iapp = []

        self.T_amb = []

        self.step_time = []
        self.step_time_mins = []
        self.test_time = []
        self.test_time_mins = []

        self.step_capacity_Ah = []


class Simulator():
    """
    Setup the Assimulo (python IDA wrapper) solver for the FULL_1D model
    defined above.
    Run the simulation for the present test schedule step.
    Store the output results in the results_holder object.
    """

    def __init__(self, conf_data, bsp_dir):
        """
        Setup the params, model, and simulator objects for the full_1d_fvm
        model.
        """
        self.bsp_dir = bsp_dir
        self.confdat = conf_data
        self.Pdat = {'RunInput': self.confdat}
        self.V_init = 4.198  # [V]

        self.Npar = conf_data['MODEL']['N_SUBMOD']
        self.Vtol = conf_data['DIST_SOLVING']['DIST_V_TOL']
        self.Kmag = conf_data['DIST_SOLVING']['K_DIST']

        self.buildpars()
        print 'Calling Tvec from simulator.__init__()'
        self.get_Tvec()

        self.buildmodel()
        self.buildsim()

    def get_Tvec(self):
        """
        Create the ambient temperature vector, indexed to the cell number in
        the multi-cell Ndist model.
        """
        try:
            if self.Npar > 1:
                self.Tvec = [(self.p.T_avg - self.p.T_dT / 2.0)
                             + i / (float(self.Npar - 1))
                             * self.p.T_dT
                             for i in range(self.Npar)]
            else:
                self.Tvec = [self.p.T_avg]
        except BaseException:
            self.Tvec = [298.15 for i in range(self.Npar)]

        print '\n'
        print '********************************'
        print '********************************'
        print 'In get_Tvec'
        print self.Tvec
        print '********************************'
        print '********************************', '\n'

    def buildpars(self):
        """
        Build the model parameter object.
        Refer to params.py for the data that self.p contains. In general,
        this contains the electrochemical parameters necessary for the
        simulation of the full Pseudo-2D Newman-style model.
        """
        self.p = params.Params()
        self.p.buildpars(self.V_init, self.Pdat)
        self.p.Ac = self.p.Area

        self.pars = self.p

    def buildmodel(self):
        """
        Setup the assimulo IDA model.
        """
        y0 = [[] for i in range(self.Npar)]
        yd0 = [[] for i in range(self.Npar)]

        print 'Tvec', self.Tvec
        for i, T in enumerate(self.Tvec):
            y0[i], yd0[i] = self.const_init_conds(T)

        # Create the model
        imp_mod = [Full_1D(self.p, y0[i], yd0[i]) for i in range(self.Npar)]

        for i, im in enumerate(imp_mod):
            im.setup_model(y0[i], yd0[i], 'test')
            im.p = self.p
            im.confdat = self.confdat

            # Sets the options to the problem
            im.algvar = [
                1.0 for i in range(
                    self.p.num_diff_vars)] + [
                0.0 for i in range(
                    self.p.num_algr_vars)]  # Set the algebraic components

        self.imp_mod = imp_mod

    def buildsim(self,):
        """
        Setup the assimulo IDA simulator.
        """
        # Create an Assimulo implicit solver (IDA)
        imp_sim = [IDA(im) for im in self.imp_mod]  # Create a IDA solver

        # Sets the paramters
        for isim in imp_sim:
            # 1e-4 #Default 1e-6
            isim.atol = self.p.RunInput['TIMESTEPPING']['SOLVER_TOL']
            # 1e-4 #Default 1e-6
            isim.rtol = self.p.RunInput['TIMESTEPPING']['SOLVER_TOL']
            # Suppres the algebraic variables on the error test
            isim.suppress_alg = True

            isim.display_progress = False
            isim.verbosity = 50
            isim.report_continuously = True
            isim.time_limit = 10.

        self.imp_sim = imp_sim

    def get_input(self, inp_typ, inp_val):
        """
        Setup the input variable for the model during the simulation based on
        the test schedule.
        """
        if inp_typ == 'Rest':
            self.inp = 0.0
            self.pars.inp_bc = 'curr'
            self.pars.rest = 1

        elif inp_typ == 'Crate':
            self.inp = (-inp_val *
                        self.confdat['MODEL']['RATE_NOM_CAP']) / self.pars.Area
            self.pars.inp_bc = 'curr'
            self.pars.rest = 0

        elif inp_typ == 'Current':
            self.inp = (-inp_val) / self.pars.Area
            self.pars.inp_bc = 'curr'
            self.pars.rest = 0

    def simulate(self, tfinal, present_step_name):
        """
        Setup the IDA model and simulator and run the present test schedule
        step.
        """
        imp_mod = self.imp_mod
        imp_sim = self.imp_sim
        p = self.p

        print 'Init T vec from y0:', [isim.y[self.p.T_ind] for isim in imp_sim]

        i_app = self.inp
        I_app = i_app * self.p.Ac
        for isim, imod in zip(imp_sim, imp_mod):
            imod.set_iapp(i_app)
        print 'I_app:', i_app * self.p.Ac, '[A]'
        print 'i_app:', i_app, '[A/m^2]'
#        i_app = I_app/self.p.Ac

        # Variable limits
        ce_lims = [1., 3990.]  # Electrolyte concentration limits

        # Simulate

        # Initialize the output variables
        t_out = []  # 0 for ts in time ]
        V_out = [[] for imod in imp_mod]  # [] # 0 for ts in time ]
        # numpy.zeros( [len(time), yb.shape[ 1]] )
        y_out = [[] for imod in imp_mod]
        # numpy.zeros( [len(time), ydb.shape[1]] )
        yd_out = [[] for imod in imp_mod]

        I_out = [[] for imod in imp_mod]

        xa_bar = [[] for imod in imp_mod]
        xc_bar = [[] for imod in imp_mod]
        Ua_bar = [[] for imod in imp_mod]
        Uc_bar = [[] for imod in imp_mod]

        csa_avg_out = [[] for imod in imp_mod]  # []
        csc_avg_out = [[] for imod in imp_mod]  # []
        Ua_avg_out = [[] for imod in imp_mod]  # []
        Uc_avg_out = [[] for imod in imp_mod]  # []

        csa_ss_out = [[] for imod in imp_mod]  # []
        csc_ss_out = [[] for imod in imp_mod]  # []
        Ua_ss_out = [[] for imod in imp_mod]  # []
        Uc_ss_out = [[] for imod in imp_mod]  # []

        eta_a_out = [[] for imod in imp_mod]  # []
        eta_c_out = [[] for imod in imp_mod]  # []
        eta_full_x = [[] for imod in imp_mod]  # []

        j_full_x = [[] for imod in imp_mod]  # []
        Uss_full_x = [[] for imod in imp_mod]  # []
        css_full_x = [[] for imod in imp_mod]  # []
        phis_full_x = [[] for imod in imp_mod]  # []

        ke_out = [[] for imod in imp_mod]  # []
        De_out = [[] for imod in imp_mod]  # []

        t_out.append(imp_sim[0].t)

        for i, isim in enumerate(imp_sim):
            imod = imp_mod[i]

            y_out[i].append(isim.y)
            yd_out[i].append(isim.yd)

            csa_now = isim.y[imod.csa_inds]  # yi[-1,imp_mod.csa_inds]
            csc_now = isim.y[imod.csc_inds]  # yi[-1,imp_mod.csc_inds]

            csa_avg_now = imod.C_cs_a_avg.dot(csa_now)
            csc_avg_now = imod.C_cs_c_avg.dot(csc_now)

            Ua_avg_now = p.uref_a(csa_avg_now / p.csa_max)
            Uc_avg_now = p.uref_c(csc_avg_now / p.csc_max)

            csa_avg_out[i].append(csa_avg_now)
            csc_avg_out[i].append(csc_avg_now)
            Ua_avg_out[i].append(Ua_avg_now)
            Uc_avg_out[i].append(Uc_avg_now)

            csa_mean = numpy.mean(imod.C_cs_a_avg.dot(csa_now))
            csc_mean = numpy.mean(imod.C_cs_c_avg.dot(csc_now))
            xa_mean = csa_mean / imod.p.csa_max
            xc_mean = csc_mean / imod.p.csc_max
            Uam = imod.p.uref_a(xa_mean)
            Ucm = imod.p.uref_c(xc_mean)

            xa_bar[i].append(xa_mean)
            xc_bar[i].append(xc_mean)
            Ua_bar[i].append(Uam)
            Uc_bar[i].append(Ucm)

            V_cell = imod.get_voltage(isim.y)

            V_out[i].append(V_cell)

            I_out[i].append(imod.i_app * p.Ac)

            ce_now = isim.y[imod.ce_inds]

            ja_now = isim.y[imod.ja_inds]
            jc_now = isim.y[imod.jc_inds]
            pa_now = isim.y[imod.pa_inds]
            pc_now = isim.y[imod.pc_inds]
            pe_now = isim.y[imod.pe_inds]

            T_now = isim.y[imod.T_ind]

            eta_a, eta_c, Uref_a_ss, Uref_c_ss, csa_ss, csc_ss \
                = imod.get_eta_uref(csa_now, csc_now, ja_now, jc_now,
                                    pa_now, pc_now, pe_now)

            eta_a_out[i].append(eta_a)
            eta_c_out[i].append(eta_c)

            eta_full_x[i].append(numpy.concatenate(
                [eta_a, numpy.zeros(imod.p.Ns), eta_c]))
            j_full_x[i].append(numpy.concatenate(
                [ja_now, numpy.zeros(imod.p.Ns), jc_now]))
            Uss_full_x[i].append(numpy.concatenate(
                [Uref_a_ss, numpy.zeros(imod.p.Ns), Uref_c_ss]))
            phis_full_x[i].append(numpy.concatenate(
                [pa_now, numpy.zeros(imod.p.Ns), pc_now]))
            css_full_x[i].append(numpy.concatenate(
                [csa_ss, numpy.zeros(imod.p.Ns), csc_ss]))

            csa_ss_out[i].append(csa_ss)
            csc_ss_out[i].append(csc_ss)

            Ua_ss_now = p.uref_a(csa_ss / p.csa_max)
            Uc_ss_now = p.uref_c(csc_ss / p.csc_max)

            Ua_ss_out[i].append(Ua_ss_now)
            Uc_ss_out[i].append(Uc_ss_now)

            ke_mid = imod.kapp_ce(ce_now, T_now, mid_on=1, eps_off=1)
            De_mid = imod.Diff_ce(ce_now, T_now, mid_on=1, eps_off=1)

            ke_out[i].append(ke_mid)
            De_out[i].append(De_mid)

        # Run two small time steps to ramp up the input current
        # This appears to be particularly important if the exchange current
        # densities for either electrode is small (i.e., at low temperatures),
        # and therefore the reaction kinetics is quite stiff.
        if i_app != 0.0:
            init_ts = numpy.linspace(0.01, 0.1, 8)
            ifact = numpy.linspace(0.01, 1, len(init_ts))**2
            for ift, dt0 in zip(ifact, numpy.gradient(init_ts)):
                for isim, imod in zip(imp_sim, imp_mod):
                    imod.set_iapp(i_app * ift)
                    isim.make_consistent('IDA_YA_YDP_INIT')
                    tb, yb, ydb = isim.simulate(isim.t + dt0, 2)
        else:
            tb = imp_sim[0].t

        # Sim out init
        V_cell = [imod.get_voltage(imp_sim[i].y)
                  for i, imod in enumerate(imp_mod)]
        ce_now = imp_sim[0].y[imp_mod[0].ce_inds]

        ti = tb

        it = 0
        print 'V_cell prior to time loop:', V_cell

        # Setup the full input current now, and initialize consistent initial
        # conditions
        for imod, isim in zip(imp_mod, imp_sim):
            imod.set_iapp(i_app)
            isim.make_consistent('IDA_YA_YDP_INIT')

        # control for the time simulation while loop
        keep_simulating = 1

        # Simulation controls
        dV_tol = self.p.RunInput['TIMESTEPPING']['DV_TOL']  # 0.02
        if i_app == 0.0:
            delta_t = self.p.delta_t_max
        else:
            delta_t = 0.01
        refined_dt = 0

        iapp_vec = numpy.array([i_app for imod in imp_mod])

        # Time sim loop
        while keep_simulating:
            # delta_t update
            if it > 1 and not refined_dt:
                dV = abs(V_out[0][-1] - V_out[0][-2])
                delta_t = dV_tol / dV * delta_t
                if delta_t > self.p.delta_t_max:
                    delta_t = self.p.delta_t_max

            # Final delta_t alignment
            if (imp_sim[0].t + delta_t) > tfinal:
                delta_t = tfinal - imp_sim[0].t + .00001

            # Run time step for each submodel, and ensure consistent voltage
            # across each submodel
            isub = 0
            Nsub = 15
            Vdiff = 2. * self.Vtol
#            Vholder = [[] for i in range(Nsub)]
#            iholder = [[] for i in range(Nsub)]

#            VerrHolder = []
#            VmeanHolder = []

#            Verrlast = numpy.zeros(self.Npar)
            t_to_sim = deepcopy(imp_sim[0].t) + delta_t
            while Vdiff > self.Vtol and isub < Nsub:
                try:
                    for isim in imp_sim:
                        ti, yi, ydi = isim.simulate(t_to_sim, 2)
                except BaseException:
                    try:
                        delta_t = delta_t * 0.1
                        if imp_sim[0].t > 0.8 * tfinal:
                            refined_dt = 1

                        for isim in imp_sim:
                            ti, yi, ydi = isim.simulate(t_to_sim, 3)
                        print '*** ran with refined delta_t ***'

                    except BaseException:
                        keep_simulating = 0
                        print 'Sim stopped due time integration failure.'

                Vnow = [im.get_voltage(imp_sim[i].y)
                        for i, im in enumerate(imp_mod)]
                Vdiff = numpy.amax(Vnow) - numpy.amin(Vnow)

                Vmean = numpy.mean(Vnow)
                Verr = numpy.array(Vnow) - Vmean

                iapp_vec = iapp_vec + Verr * self.Kmag

                for i, imod in enumerate(imp_mod):
                    imod.set_iapp(iapp_vec[i])
                    imp_sim[i].make_consistent('IDA_YA_YDP_INIT')

                isub += 1

            # Update the outputs for each submodel
            t_out.append(imp_sim[0].t)

            for i, isim in enumerate(imp_sim):
                imod = imp_mod[i]

                y_out[i].append(isim.y)
                yd_out[i].append(isim.yd)

                csa_now = isim.y[imod.csa_inds]  # yi[-1,imp_mod.csa_inds]
                csc_now = isim.y[imod.csc_inds]  # yi[-1,imp_mod.csc_inds]

                csa_avg_now = imod.C_cs_a_avg.dot(csa_now)
                csc_avg_now = imod.C_cs_c_avg.dot(csc_now)

                Ua_avg_now = p.uref_a(csa_avg_now / p.csa_max)
                Uc_avg_now = p.uref_c(csc_avg_now / p.csc_max)

                csa_avg_out[i].append(csa_avg_now)
                csc_avg_out[i].append(csc_avg_now)
                Ua_avg_out[i].append(Ua_avg_now)
                Uc_avg_out[i].append(Uc_avg_now)

                csa_mean = numpy.mean(imod.C_cs_a_avg.dot(csa_now))
                csc_mean = numpy.mean(imod.C_cs_c_avg.dot(csc_now))
                xa_mean = csa_mean / imod.p.csa_max
                xc_mean = csc_mean / imod.p.csc_max
                Uam = imod.p.uref_a(xa_mean)
                Ucm = imod.p.uref_c(xc_mean)

                xa_bar[i].append(xa_mean)
                xc_bar[i].append(xc_mean)
                Ua_bar[i].append(Uam)
                Uc_bar[i].append(Ucm)

                V_cell = imod.get_voltage(isim.y)

                V_out[i].append(V_cell)

                I_out[i].append(imod.i_app * p.Ac)

                ce_now = isim.y[imod.ce_inds]

                ja_now = isim.y[imod.ja_inds]
                jc_now = isim.y[imod.jc_inds]
                pa_now = isim.y[imod.pa_inds]
                pc_now = isim.y[imod.pc_inds]
                pe_now = isim.y[imod.pe_inds]

                T_now = isim.y[imod.T_ind]

                eta_a, eta_c, Uref_a_ss, Uref_c_ss, csa_ss, csc_ss \
                    = imod.get_eta_uref(csa_now, csc_now, ja_now, jc_now,
                                        pa_now, pc_now, pe_now)

                eta_a_out[i].append(eta_a)
                eta_c_out[i].append(eta_c)

                eta_full_x[i].append(numpy.concatenate(
                    [eta_a, numpy.zeros(imod.p.Ns), eta_c]))
                j_full_x[i].append(numpy.concatenate(
                    [ja_now, numpy.zeros(imod.p.Ns), jc_now]))
                Uss_full_x[i].append(numpy.concatenate(
                    [Uref_a_ss, numpy.zeros(imod.p.Ns), Uref_c_ss]))
                phis_full_x[i].append(numpy.concatenate(
                    [pa_now, numpy.zeros(imod.p.Ns), pc_now]))
                css_full_x[i].append(numpy.concatenate(
                    [csa_ss, numpy.zeros(imod.p.Ns), csc_ss]))

                csa_ss_out[i].append(csa_ss)
                csc_ss_out[i].append(csc_ss)

                Ua_ss_now = p.uref_a(csa_ss / p.csa_max)
                Uc_ss_now = p.uref_c(csc_ss / p.csc_max)

                Ua_ss_out[i].append(Ua_ss_now)
                Uc_ss_out[i].append(Uc_ss_now)

                ke_mid = imod.kapp_ce(ce_now, T_now, mid_on=1, eps_off=1)
                De_mid = imod.Diff_ce(ce_now, T_now, mid_on=1, eps_off=1)

                ke_out[i].append(ke_mid)
                De_out[i].append(De_mid)

            print 'time:', round(imp_sim[0].t, 3), ' |  Voltage:', \
                round(V_cell, 3), ' |  Current:', \
                round(i_app * self.pars.Ac, 3), ' |  ' \
                + str(round(imp_sim[0].t / tfinal * 100., 1)) \
                + '% complete  |  delta_t:', round(delta_t, 3), \
                ' |  refined_dt:', refined_dt, ' |  Vdiff:', Vdiff

            # Check sim stop limits
            # Cell voltage
            if V_cell <= p.volt_min:
                print '\n', 'Vmin stopped simulation.'
                keep_simulating = 0
            elif V_cell >= p.volt_max:
                print '\n', 'Vmax stopped simulation.'
                keep_simulating = 0
            # E-lyte concentration saturation
            elif max(ce_now) >= max(ce_lims):
                print '\n', 'ce max stopped simulation.'
                keep_simulating = 0
            elif min(ce_now) <= min(ce_lims):
                print '\n', 'ce min stopped simulation.'
                keep_simulating = 0
            # Sim time stop
            elif imp_sim[0].t >= tfinal:
                keep_simulating = 0
            # Anode surface equilibrium potential
            elif max(Ua_ss_now) >= p.an_volt_max:
                keep_simulating = 0
                print '\n', 'Ua_ss max stopped simulation.'
            elif min(Ua_ss_now) <= p.an_volt_min:
                keep_simulating = 0
                print '\n', 'Ua_ss min stopped simulation.'
            # Cathode surface equilibrium potential
            elif max(Uc_ss_now) >= p.cat_volt_max:
                keep_simulating = 0
                print '\n', 'Uc_ss max stopped simulation.'
            elif min(Uc_ss_now) <= p.cat_volt_min:
                keep_simulating = 0
                print '\n', 'Uc_ss min stopped simulation.'

            it += 1

        # Prepare the final output variables
        y1 = [numpy.array(y_outi) for y_outi in y_out]

        states = {}
        states['c_s_a'] = [y1[i][:, p.csa_inds] for i in range(self.Npar)]
        states['c_s_c'] = [y1[i][:, p.csc_inds] for i in range(self.Npar)]
        states['c_e'] = [y1[i][:, p.ce_inds] for i in range(self.Npar)]
        states['T'] = [y1[i][:, p.T_ind] for i in range(self.Npar)]

        states['phi_e'] = [y1[i][:, p.pe_inds] for i in range(self.Npar)]
        states['phi_s_a'] = [y1[i][:, p.pa_inds] for i in range(self.Npar)]
        states['phi_s_c'] = [y1[i][:, p.pc_inds] for i in range(self.Npar)]

        states['ja'] = [y1[i][:, p.ja_inds] for i in range(self.Npar)]
        states['jc'] = [y1[i][:, p.jc_inds] for i in range(self.Npar)]

#        pa_cc = states['phi_s_a'][:, 0]
#        pc_cc = states['phi_s_c'][:,-1]
#        pe_midsep = states['phi_e'][:,int(p.Na+(p.Ns/2.))]

        Va = [states['phi_s_a'][i][:, 0]
              - states['phi_e'][i][:, int(p.Na + (p.Ns / 2.))]
              for i in range(self.Npar)]  # pa_cc - pe_midsep
        Vc = [states['phi_s_c'][i][:, -1]
              - states['phi_e'][i][:, int(p.Na + (p.Ns / 2.))]
              for i in range(self.Npar)]  # pc_cc - pe_midsep

        mergExtr = {}
        mergExtr['Volt'] = [numpy.array(V_out[i]) for i in range(self.Npar)]
        mergExtr['Cur'] = [numpy.array(I_out[i]) for i in range(self.Npar)]
        mergExtr['Iapp'] = I_app

        mergExtr['Va'] = Va
        mergExtr['Vc'] = Vc
        mergExtr['step_time'] = numpy.array(t_out) - t_out[0]
        mergExtr['step_time_mins'] = mergExtr['step_time'] / 60.

        mergExtr['test_time'] = numpy.array(t_out)
        mergExtr['test_time_mins'] = mergExtr['test_time'] / 60.

        mergExtr['step_capacity_Ah'] = [
            scipy.integrate.cumtrapz(
                mergExtr['Cur'][i],
                x=mergExtr['step_time'] / 3600.,
                initial=0.0) for i in range(self.Npar)]

        mergExtr['css_fullx'] = [numpy.array(css_full_x[i])
                                 for i in range(self.Npar)]
        mergExtr['Uss_fullx'] = [numpy.array(Uss_full_x[i])
                                 for i in range(self.Npar)]
        mergExtr['phis_fullx'] = [numpy.array(phis_full_x[i])
                                  for i in range(self.Npar)]
        mergExtr['eta_fullx'] = [numpy.array(eta_full_x[i])
                                 for i in range(self.Npar)]
        mergExtr['j_fullx'] = [numpy.array(j_full_x[i])
                               for i in range(self.Npar)]

        mergExtr['eta_a'] = [numpy.array(eta_a_out[i])
                             for i in range(self.Npar)]
        mergExtr['eta_c'] = [numpy.array(eta_c_out[i])
                             for i in range(self.Npar)]

        mergExtr['csa_avg'] = [numpy.array(csa_avg_out[i])
                               for i in range(self.Npar)]
        mergExtr['csc_avg'] = [numpy.array(csc_avg_out[i])
                               for i in range(self.Npar)]
        mergExtr['csa_ss'] = [numpy.array(csa_ss_out[i])
                              for i in range(self.Npar)]
        mergExtr['csc_ss'] = [numpy.array(csc_ss_out[i])
                              for i in range(self.Npar)]

        mergExtr['Ua_avg'] = [numpy.array(Ua_avg_out[i])
                              for i in range(self.Npar)]
        mergExtr['Uc_avg'] = [numpy.array(Uc_avg_out[i])
                              for i in range(self.Npar)]
        mergExtr['Ua_ss'] = [numpy.array(Ua_ss_out[i])
                             for i in range(self.Npar)]
        mergExtr['Uc_ss'] = [numpy.array(Uc_ss_out[i])
                             for i in range(self.Npar)]

        mergExtr['ke'] = [numpy.array(ke_out[i]) for i in range(self.Npar)]
        mergExtr['De'] = [numpy.array(De_out[i]) for i in range(self.Npar)]

        self.t_end_now = imp_sim[0].t

        # Assign the desired output variables to results holder object
        self.assign_model_results(states, mergExtr, present_step_name)

    def const_init_conds(self, T):
        """
        For the first step in a schedule, setup the initial conditions, where
        the spatial states are uniform. i.e,
        c_s_n = csn0*numpy.zeros( p.SolidOrder_n*p.Nn ) and csn0 is a scalar
        and equal to p.theta_n0*p.c_s_n_max
        """
        p = self.pars
        y0 = numpy.zeros(p.num_diff_vars + p.num_algr_vars)

        # x0
        y0[p.ce_inds] = p.ce_0 * numpy.ones(p.N, dtype='d')
        y0[p.csa_inds] = (p.theta_a0 * p.csa_max) * \
            numpy.ones(p.Na * p.Nra, dtype='d')
        y0[p.csc_inds] = (p.theta_c0 * p.csc_max) * \
            numpy.ones(p.Nc * p.Nrc, dtype='d')
        y0[p.T_ind] = deepcopy(T)

        # z0
        y0[p.pa_inds] = p.uref_a(p.theta_a0 * numpy.ones(p.Na, dtype='d'))
        y0[p.pc_inds] = p.uref_c(p.theta_c0 * numpy.ones(p.Nc, dtype='d'))
        y0[p.pe_inds] = numpy.zeros(p.N, dtype='d')
        y0[p.ja_inds] = numpy.zeros(p.Na, dtype='d')
        y0[p.jc_inds] = numpy.zeros(p.Nc, dtype='d')

        return y0, numpy.zeros_like(y0)

    def build_results_dict(self, steps, cycs):
        """
        Dict data structure for the simulation results.
        Keys use the following form: 'stepX_repeatY', where X and Y are the
        step and cycle numbers, respectively.
        """
        self.results_out = dict([('step' + str(stp) + '_repeat' + str(cyc),
                                  Results_object(self.pars))
                                 for stp in range(steps)
                                 for cyc in range(cycs)])

    def assign_model_results(self, states, extras, present_run):
        """
        Take the final solution from cn_solver and assign the values to the
        results dict.
        """
        self.assign_dict_to_class(states, self.results_out[present_run])
        self.assign_dict_to_class(extras, self.results_out[present_run])

    def assign_dict_to_class(self, dict_obj, class_obj):
        """
        Take the dict values and assign to results class vars.
        """
        for key, val in dict_obj.iteritems():
            if hasattr(class_obj, key):
                setattr(class_obj, key, val)
