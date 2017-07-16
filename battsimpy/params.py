# -*- coding: utf-8 -*-
"""Model parameter class.
"""
import numpy
import scipy.interpolate
# from matplotlib import pyplot as plt

# battsimpy specific modules
import confreader
import batteqns


class Params():
    """
    Contains all parameters for the battery model.
    """

    def __init__(self):
        self.pi = 3.14159265358

    def buildpars(self, V_init, Pdat):
        """
        Parameter builder for all models in BattSimPy
        """
        self.V_init = V_init
        self.Pdat = Pdat

        self.RunInput = Pdat['RunInput']
        RunInput = Pdat['RunInput']
        fname_root = RunInput['FILEPATHS']['DATA_ROOT'] + \
            'Model_' + RunInput['FILEPATHS']['MODEL_NAME'] + '/' + \
            RunInput['FILEPATHS']['PARAMS'] + '/'

        # Check if physics type model
        if 'ecm' in RunInput['MODEL']['MODEL_TYPE'] \
                or 'ECM' in RunInput['MODEL']['MODEL_TYPE']:
            self.modelPhysical = 0
        else:
            self.modelPhysical = 1

        # Universal constants
        self.F = 96487.0  # [Coulombs/mol]
        self.R_gas = 8.314472  # [J/(mol-K)]

        # T used internally here for some initilization of parameters
#        T_init = 298.15 # [K]
        self.T_amb = RunInput['THERMAL']['T_AMBIENT']

        # Parameters unique to physics based models
        if self.modelPhysical:

            # io optimization boolean control variables
            self.kn_opt_on = RunInput['OPTIMIZATION']['ION_OPT_ON']
            self.kp_opt_on = RunInput['OPTIMIZATION']['IOP_OPT_ON']

            # Solid phase transport (Ds) parameters
            self.Dsdat_p = {
                'stoich_sens_on':
                RunInput['SOLID_DIFFUSION']['VAR_DIFF_CATHODE_ON']}
            self.Dsdat_n = {
                'stoich_sens_on':
                RunInput['SOLID_DIFFUSION']['VAR_DIFF_ANODE_ON']}

            # Radial Ds sensitivity control
            # (e.g., If Ds is lower at particle surface)
#            self.Dsdat_n['r_sens_on'] \
#                = RunInput['SOLID_DIFFUSION']['R_DIFF_ANODE_ON']
#            self.Dsdat_p['r_sens_on'] \
#                = RunInput['SOLID_DIFFUSION']['R_DIFF_CATHODE_ON']
#            if self.Dsdat_p['r_sens_on'] :
#                self.Dsdat_p['DsR'] = {'r':[], 'coeff':[]}
#                DsR_cat = numpy.loadtxt(fname_root
#                                        +'solid/diffusion/'+'DsR_cathode.csv',
#                                        dtype='d', delimiter=',')
#                self.Dsdat_p['r'] = DsR_cat[:,0].flatten()
#                self.Dsdat_p['coeff'] = DsR_cat[:,1].flatten()

#            if self.Dsdat_n['r_sens_on'] :
#                self.Dsdat_n['DsR'] = {'r':[], 'coeff':[]}
#                DsR_an = numpy.loadtxt(fname_root
#                                       + 'solid/diffusion/'
#                                       +'DsR_anode.csv',
#                                       dtype='d', delimiter=',')
#                self.Dsdat_n['r'] = DsR_an[:,0].flatten()
#                self.Dsdat_n['coeff'] = DsR_an[:,1].flatten()

            # Discretization through sandwich thickness

            # 1D linear mesh node parameters
            # Number of points in the x dimension for anode
            Na = int(RunInput['MESH']['NA'])
            # Number of points in the x dimension for separator
            Ns = int(RunInput['MESH']['NS'])
            # Number of points in the x dimension for cathode
            Nc = int(RunInput['MESH']['NC'])
            N = Na + Ns + Nc

            self.Na = Na
            self.Ns = Ns
            self.Nc = Nc
            self.N = N

            # Radial mesh node parameters
            # Number of points in the anode particle dim
            Nra = int(RunInput['MESH']['NRA'])
            # Number of points in the cathode particle dim
            Nrc = int(RunInput['MESH']['NRC'])
            self.Nra = Nra
            self.Nrc = Nrc

            self.num_diff_vars = (self.N
                                  + self.Nra * self.Na
                                  + self.Nrc * self.Nc + 1)
            self.num_algr_vars = (self.Na
                                  + self.Nc + self.N + self.Na + self.Nc)

            # Material properties
            self.get_matl_properties(fname_root + 'matl_prop.txt')
            self.ce_nom = self.matl_prop['ELECTROLYTE']['c_e_ref']

            self.ce_0 = RunInput['ELECTROLYTE']['C_E_INIT']

            # Design properties
            self.get_des_properties(fname_root + 'des_prop.txt')

            # x mesh
            self.X = self.La + self.Ls + self.Lc
            La, Ls, Lc = self.La, self.Ls, self.Lc

            self.x_e = numpy.array(list(numpy.linspace(0.0, La, Na + 1))
                                   + list(numpy.linspace(La + Ls / float(Ns),
                                                         (Ls + La
                                                          - Ls / float(Ns)),
                                                         Ns - 1))
                                   + list(numpy.linspace(La + Ls,
                                                         La + Ls + Lc,
                                                         Nc + 1)))
            self.x_m = numpy.array([0.5
                                    * (self.x_e[i + 1] + self.x_e[i])
                                    for i in range(N)], dtype='d')
            self.vols = numpy.array([(self.x_e[i + 1] - self.x_e[i])
                                    for i in range(N)], dtype='d')

            # Particle discretization variables
            self.cstype = RunInput['MESH']['CS_TYPE']
            Wa = RunInput['MESH']['CS_WA']
            Wc = RunInput['MESH']['CS_WC']

            self.r_e_a = batteqns.nonlinspace(self.Rp_a, Wa, Nra + 1)
            self.r_m_a = numpy.array([0.5
                                     * (self.r_e_a[i + 1]
                                        + self.r_e_a[i])
                                     for i in range(Nra)], dtype='d')
            self.r_e_c = batteqns.nonlinspace(self.Rp_c, Wc, Nrc + 1)
            self.r_m_c = numpy.array([0.5
                                     * (self.r_e_c[i + 1]
                                        + self.r_e_c[i])
                                     for i in range(Nrc)], dtype='d')
            self.vols_ra_m = numpy.array([1/3.
                                         * (self.r_e_a[i + 1]**3
                                            - self.r_e_a[i]**3)
                                         for i in range(Nra)], dtype='d')
            self.vols_rc_m = numpy.array([1/3.
                                         * (self.r_e_c[i + 1]**3
                                            - self.r_e_c[i]**3)
                                         for i in range(Nrc)], dtype='d')

            # Useful sub-meshes for the phi_s functions
            self.x_m_a = self.x_m[:Na]
            self.x_m_c = self.x_m[-Nc:]
            self.x_e_a = self.x_e[:Na + 1]
            self.x_e_c = self.x_e[-Nc - 1:]

            self.vols_a = self.vols[:Na]
            self.vols_c = self.vols[-Nc:]

            self.as_a_mean = 1. / self.La * \
                sum([asa*v for asa, v in zip(self.as_a, self.vols[:self.Na])])
            self.as_c_mean = 1. / self.Lc * \
                sum([asc*v for asc, v in zip(self.as_c, self.vols[-self.Nc:])])

            # Tab and foil ohmic resistances
            self.Rfl = RunInput['MODEL']['FOIL_RES']  # [Ohms]
            self.Rtb = RunInput['MODEL']['TAB_RES']  # [Ohms]
            # Cathode ohmic resistance used for single particle model
            self.Rocat = 0.0  # [Ohms]

            # Total active electrochemical area
            self.S_n = self.as_a_mean * self.La * self.Area
            self.S_p = self.as_c_mean * self.Lc * self.Area

            # PSD stuff (not yet)

            # Active material electronic conductivity
            # [1/Ohm-m]
            self.sig_a_eff = self.matl_prop['SOLID_CONDUCTIVITY']['sig_a'] * \
                self.eps_s_a_eff
            # [1/Ohm-m]
            self.sig_c_eff = self.matl_prop['SOLID_CONDUCTIVITY']['sig_c'] * \
                self.eps_s_c_eff

            # Electrolyte transport properties
            self.activ_on = RunInput['ELECTROLYTE']['ACTIVITY_ON']
            De_fn = fname_root + 'electrolyte/' + \
                RunInput['ELECTROLYTE']['DE_FN']
            kappa_fn = fname_root + 'electrolyte/' + \
                RunInput['ELECTROLYTE']['KAP_FN']
#            fca_fn   = fname_root + 'electrolyte/' \
#                       + RunInput['ELECTROLYTE']['FCA_FN']

            # Interpolators for De, ke
            print "RunInput['ELECTROLYTE']['DE_FACTOR']", \
                  RunInput['ELECTROLYTE']['DE_FACTOR']
            self.De_intp, ce_lims_De = batteqns.build_interp_2d(
                De_fn, scalar=RunInput['ELECTROLYTE']['DE_FACTOR'])
            self.ke_intp, ce_lims_ke = batteqns.build_interp_2d(
                kappa_fn, scalar=RunInput['ELECTROLYTE']['KE_FACTOR'])

            if (ce_lims_De[0] >= ce_lims_ke[0]) and (
                    ce_lims_De[1] <= ce_lims_ke[1]):
                self.ce_lims = ce_lims_De
            else:
                self.ce_lims = ce_lims_ke

            self.R_ke = RunInput['ELECTROLYTE']['RKE']
            self.Ea_ke = RunInput['ELECTROLYTE']['EA_KE']

            # Electrolyte reference concentration
            self.c_e_ref = 1000.  # [mol/m^3]

            # --- Kinetic paramters --- #
            ioa_fn = fname_root + 'solid/kinetics/' + \
                RunInput['KINETICS']['IOA_FN']
            ioc_fn = fname_root + 'solid/kinetics/' + \
                RunInput['KINETICS']['IOC_FN']

            self.ioa_coeff = RunInput['KINETICS']['IOA_COEF']
            self.ioc_coeff = RunInput['KINETICS']['IOC_COEF']

            self.ioa_Ea = RunInput['KINETICS']['IOA_EA']
            self.ioc_Ea = RunInput['KINETICS']['IOC_EA']

            self.ioa_const = RunInput['KINETICS']['IOA_CONST']
            self.ioc_const = RunInput['KINETICS']['IOC_CONST']

            self.ioa_interp, junk = batteqns.build_interp_2d(ioa_fn)
            self.ioc_interp, junk = batteqns.build_interp_2d(ioc_fn)

#            self.io_a = 10.0 # [A/m^2]
#            self.io_c = 10.0 # [A/m^2]

            # Film resistance parameter loading
            # (TODO: determine the film_res file format and load method)
#            Rf_dat = conf_reader.reader( fname_root
#                                          + '/' + RunInput['film_res.csv'] )
            self.Rf_a = numpy.zeros(self.Na)  # Rf_dat['R_f_n']   # [Ohms-m^2]
            self.Rf_c = numpy.zeros(self.Nc)  # Rf_dat['R_f_p']   # [Ohms-m^2]
            self.Rfa_Ea = 0.  # Rf_dat['Rfn_Ea']  # [Ohms-m^2]
            self.Rfc_Ea = 0.  # Rf_dat['Rfp_Ea']  # [Ohms-m^2]

            # Double layer capacitance (TODO: add SPM_dl)
            # self.Cdl_n = 0.2
            # self.Cdl_p = 1.2
            # self.Cdl_kmm = 1.2

            # Equilibrium potentials
            # 'solid/thermodynamics/Un_simba_20170512.csv'
            Ua_fn = fname_root + 'solid/thermodynamics/' + \
                RunInput['THERMODYNAMIC']['AN_POTENTIAL_FN']
            # 'solid/thermodynamics/Up_simba_20170512.csv'
            Uc_fn = fname_root + 'solid/thermodynamics/' + \
                RunInput['THERMODYNAMIC']['CAT_POTENTIAL_FN']

            self.uref_a, self.uref_c, self.duref_a, self.duref_c \
                = batteqns.get_smooth_Uref_data(Ua_fn, Uc_fn, filter_on=0)

            # --- Ds, solid mass transport --- #
            self.Ea_Dsa = RunInput['SOLID_DIFFUSION']['Dsa_Ea']
            self.Ea_Dsc = RunInput['SOLID_DIFFUSION']['Dsc_Ea']

            Dsa_fn = fname_root + 'solid/diffusion/' + \
                RunInput['SOLID_DIFFUSION']['DSA_FN']
            Dsc_fn = fname_root + 'solid/diffusion/' + \
                RunInput['SOLID_DIFFUSION']['DSC_FN']

            Dsa_map = numpy.loadtxt(Dsa_fn, delimiter=",")
            Dsc_map = numpy.loadtxt(Dsc_fn, delimiter=",")

            if Dsa_map[1, 0] < Dsa_map[0, 0]:
                Dsa_map = numpy.flipud(Dsa_map)
            if Dsc_map[1, 0] < Dsc_map[0, 0]:
                Dsc_map = numpy.flipud(Dsc_map)

            Dsc_map[:, 1] = RunInput['SOLID_DIFFUSION']['Dsc_coeff'] * \
                Dsc_map[:, 1]

            # Create the interpolators
            self.Dsa_intp = scipy.interpolate.interp1d(
                Dsa_map[:, 0], Dsa_map[:, 1], kind='linear',
                fill_value=Dsa_map[-1, 1], bounds_error=False)
            self.Dsc_intp = scipy.interpolate.interp1d(
                Dsc_map[:, 0], Dsc_map[:, 1], kind='linear',
                fill_value=Dsc_map[-1, 1], bounds_error=False)

            Dsa = numpy.mean(Dsa_map[:, 1])
            Dsc = numpy.mean(Dsc_map[:, 1])
            self.Dsa = Dsa
            self.Dsc = Dsc

            # Thermal model parameters
            # conv heat coeff [W/m^2-K]
            self.h = RunInput['THERMAL']['H_CONV']
            # conv area ratio (Aconv/Acoat) [m^2/m^2]
            self.Aconv = RunInput['THERMAL']['A_CONV_RATIO']
            self.rho = RunInput['THERMAL']['CELL_DENSITY'] / \
                self.X  # density per coated area [kg/m^2]
            # specific heat capacity [J/kg-K]
            self.Cp = RunInput['THERMAL']['CELL_SPECIFIC_HEAT']

            # Initial stoichs for anode and cathode
            self.get_init_thetas(V_init,
                                 RunInput['THERMODYNAMIC']['THETA_A_TOP'],
                                 RunInput['THERMODYNAMIC']['THETA_C_TOP'])

            # System indices
            self.ce_inds = range(self.N)
            self.ce_inds_r = numpy.reshape(
                self.ce_inds, [len(self.ce_inds), 1])
            self.ce_inds_c = numpy.reshape(
                self.ce_inds, [1, len(self.ce_inds)])

            self.csa_inds = range(self.N, self.N + (self.Na * self.Nra))
            self.csa_inds_r = numpy.reshape(
                self.csa_inds, [len(self.csa_inds), 1])
            self.csa_inds_c = numpy.reshape(
                self.csa_inds, [1, len(self.csa_inds)])

            self.csc_inds = range(self.N + (self.Na * self.Nra),
                                  self.N + (self.Na * self.Nra)
                                  + (self.Nc * self.Nrc))
            self.csc_inds_r = numpy.reshape(
                self.csc_inds, [len(self.csc_inds), 1])
            self.csc_inds_c = numpy.reshape(
                self.csc_inds, [1, len(self.csc_inds)])

            self.T_ind = self.N + (self.Na * self.Nra) + (self.Nc * self.Nrc)

            c_end = self.N + (self.Na * self.Nra) + (self.Nc * self.Nrc) + 1

            self.ja_inds = range(c_end, c_end + self.Na)
            self.ja_inds_r = numpy.reshape(
                self.ja_inds, [len(self.ja_inds), 1])
            self.ja_inds_c = numpy.reshape(
                self.ja_inds, [1, len(self.ja_inds)])

            self.jc_inds = range(c_end + self.Na, c_end + self.Na + self.Nc)
            self.jc_inds_r = numpy.reshape(
                self.jc_inds, [len(self.jc_inds), 1])
            self.jc_inds_c = numpy.reshape(
                self.jc_inds, [1, len(self.jc_inds)])

            self.pe_inds = range(
                c_end + self.Na + self.Nc,
                c_end + self.Na + self.Nc + self.N)
            self.pe_inds_r = numpy.reshape(
                self.pe_inds, [len(self.pe_inds), 1])
            self.pe_inds_c = numpy.reshape(
                self.pe_inds, [1, len(self.pe_inds)])

            self.pe_a_inds = range(
                c_end + self.Na + self.Nc,
                c_end + self.Na + self.Nc + self.Na)
            self.pe_a_inds_r = numpy.reshape(
                self.pe_a_inds, [len(self.pe_a_inds), 1])
            self.pe_a_inds_c = numpy.reshape(
                self.pe_a_inds, [1, len(self.pe_a_inds)])

            self.pe_c_inds = range(
                c_end + self.Na + self.Nc + self.Na + self.Ns,
                c_end + self.Na + self.Nc + self.N)
            self.pe_c_inds_r = numpy.reshape(
                self.pe_c_inds, [len(self.pe_c_inds), 1])
            self.pe_c_inds_c = numpy.reshape(
                self.pe_c_inds, [1, len(self.pe_c_inds)])

            self.pa_inds = range(
                c_end + self.Na + self.Nc + self.N,
                c_end + self.Na + self.Nc + self.N + self.Na)
            self.pa_inds_r = numpy.reshape(
                self.pa_inds, [len(self.pa_inds), 1])
            self.pa_inds_c = numpy.reshape(
                self.pa_inds, [1, len(self.pa_inds)])

            self.pc_inds = range(
                c_end +
                self.Na +
                self.Nc +
                self.N +
                self.Na,
                c_end +
                self.Na +
                self.Nc +
                self.N +
                self.Na +
                self.Nc)
            self.pc_inds_r = numpy.reshape(
                self.pc_inds, [len(self.pc_inds), 1])
            self.pc_inds_c = numpy.reshape(
                self.pc_inds, [1, len(self.pc_inds)])

            # second set for manual jac version
            c_end = 0
            self.ja_inds2 = range(c_end, c_end + self.Na)
            self.ja_inds_r2 = numpy.reshape(
                self.ja_inds2, [len(self.ja_inds2), 1])
            self.ja_inds_c2 = numpy.reshape(
                self.ja_inds2, [1, len(self.ja_inds2)])

            self.jc_inds2 = range(c_end + self.Na, c_end + self.Na + self.Nc)
            self.jc_inds_r2 = numpy.reshape(
                self.jc_inds2, [len(self.jc_inds2), 1])
            self.jc_inds_c2 = numpy.reshape(
                self.jc_inds2, [1, len(self.jc_inds2)])

            self.pe_inds2 = range(
                c_end + self.Na + self.Nc,
                c_end + self.Na + self.Nc + self.N)
            self.pe_inds_r2 = numpy.reshape(
                self.pe_inds2, [len(self.pe_inds2), 1])
            self.pe_inds_c2 = numpy.reshape(
                self.pe_inds2, [1, len(self.pe_inds2)])

            self.pe_a_inds2 = range(
                c_end + self.Na + self.Nc,
                c_end + self.Na + self.Nc + self.Na)
            self.pe_a_inds_r2 = numpy.reshape(
                self.pe_a_inds2, [len(self.pe_a_inds2), 1])
            self.pe_a_inds_c2 = numpy.reshape(
                self.pe_a_inds2, [1, len(self.pe_a_inds2)])

            self.pe_c_inds2 = range(
                c_end + self.Na + self.Nc + self.Na + self.Ns,
                c_end + self.Na + self.Nc + self.N)
            self.pe_c_inds_r2 = numpy.reshape(
                self.pe_c_inds2, [len(self.pe_c_inds2), 1])
            self.pe_c_inds_c2 = numpy.reshape(
                self.pe_c_inds2, [1, len(self.pe_c_inds2)])

            self.pa_inds2 = range(
                c_end + self.Na + self.Nc + self.N,
                c_end + self.Na + self.Nc + self.N + self.Na)
            self.pa_inds_r2 = numpy.reshape(
                self.pa_inds2, [len(self.pa_inds2), 1])
            self.pa_inds_c2 = numpy.reshape(
                self.pa_inds2, [1, len(self.pa_inds2)])

            self.pc_inds2 = range(
                c_end +
                self.Na +
                self.Nc +
                self.N +
                self.Na,
                c_end +
                self.Na +
                self.Nc +
                self.N +
                self.Na +
                self.Nc)
            self.pc_inds_r2 = numpy.reshape(
                self.pc_inds2, [len(self.pc_inds2), 1])
            self.pc_inds_c2 = numpy.reshape(
                self.pc_inds2, [1, len(self.pc_inds2)])

        else:
            self.ecmOrder = RunInput['ECM_PARAMS']['ECM_ORDER']
            self.ecm_params = {'ocv': {'intp_func': [], 'dim': ''},
                               'res_ohm': {'intp_func': [], 'dim': ''},
                               'res': [{'intp_func': [], 'dim':''}
                                       for i_rc in range(self.ecmOrder)],
                               'tau': [{'intp_func': [], 'dim':''}
                                       for i_rc in range(self.ecmOrder)], }
            # OCV data
            self.ocv_fn = fname_root + 'ecm/ocv/' + \
                self.RunInput['ECM_PARAMS']['OCV_FNAME']
            par_out = self.gen_param_interp_function(self.ocv_fn)
            if par_out[1] == '1D':
                self.ecm_params['ocv']['intp_func'], \
                    self.ecm_params['ocv']['dim'], d, d1 = par_out
                self.OCP = d
            elif par_out[1] == '2D':
                self.ecm_params['ocv']['intp_func'], \
                    self.ecm_params['ocv']['dim'], d, D1, D2, d1, d2 = par_out
                self.OCP = d[:, 0]
            self.xs = d1

            # Ohmic Resistance data
            self.res_ohm_fn = fname_root + 'ecm/res/' + \
                self.RunInput['ECM_PARAMS']['RES_OHMIC_FNAME']
            par_out = self.gen_param_interp_function(self.res_ohm_fn)
            if par_out[1] == '1D':
                self.ecm_params['res_ohm']['intp_func'], \
                    self.ecm_params['res_ohm']['dim'], d, d1 = par_out
            elif par_out[1] == '2D':
                self.ecm_params['res_ohm']['intp_func'], \
                    self.ecm_params['res_ohm']['dim'], d, D1, D2, \
                    d1, d2 = par_out

            # Loop through for all RC elements
            self.res_fn = [fname_root + 'ecm/res/' +
                           self.RunInput['ECM_PARAMS']['RES_RC_FNAMES'][i_rc]
                           for i_rc in range(self.ecmOrder)]
            self.tau_fn = [fname_root + 'ecm/tau/' +
                           self.RunInput['ECM_PARAMS']['TAU_RC_FNAMES'][i_rc]
                           for i_rc in range(self.ecmOrder)]
            for i_rc in range(self.ecmOrder):
                # Resistance data
                par_out = self.gen_param_interp_function(self.res_fn[i_rc])
                self.ecm_params['res'][i_rc]['intp_func'] = par_out[0]
                self.ecm_params['res'][i_rc]['dim'] = par_out[1]

                # Time constant (tau) data
                par_out = self.gen_param_interp_function(self.tau_fn[i_rc])
                self.ecm_params['tau'][i_rc]['intp_func'] = par_out[0]
                self.ecm_params['tau'][i_rc]['dim'] = par_out[1]

            # Cell capacity
            self.ecm_params['Ah_cap'] = RunInput['ECM_PARAMS']['ECM_OCV_CAP']

            # Initial state of charge
            self.SOC_0 = numpy.interp(self.V_init, self.OCP, self.xs)

        # KMM parameters for distributed models
        self.Ksm = RunInput['DIST_SOLVING']['K_DIST']
        self.sm_iter = RunInput['DIST_SOLVING']['SM_ITER_MAX']
        self.V_err_tol = RunInput['DIST_SOLVING']['DIST_V_TOL']
        self.I_kmm_dist = 1e-3
#        self.C_kmm_dist = 1e-3
        self.n_submod = RunInput['MODEL']['N_SUBMOD']

        # --- Crank-Nicholson Control --- #
        self.max_rest_step = RunInput['TIMESTEPPING']['MAX_REST_STEP']

    def gen_param_interp_function(self, fpath, x_scale=1.0, y_scale=1.0,
                                  z_scale=1.0):
        """
        Create a 1D or 2D interpolation function for the file containing a data
        table.
        The following .csv file table format is required:
        0 ,Y1 ,Y2 ,Y3 ,...YN
        X1,P11,P12,P13,...P1N
        X2,P21,P22,P23,...P2N
        X3,P31,P32,P33,...P3N
        .
        .
        .
        XM,PM1,PM2,PM3,...PMN

        Xi  -> indicates the first dimension of the interpolation.
        Tj  -> indicates the second dimension of the interpolation.
        Pij -> indicates the parameter at the Xi,Yj point in the map.

        Numbers should be the only values used in any element. This is
        constrained by us using numpy.loadtxt to load the data table file.
        """
        # Load the data table
        data_table = numpy.loadtxt(fpath, dtype='d', delimiter=',')

        # Check for dimensions (handle either 1D or 2D interpolation)
        if len(data_table.shape) > 1:   # Ensure more than one column used
            # multiple Y values used (2D interpolation)
            if data_table.shape[1] > 2:
                d1 = data_table[1:, 0] * x_scale
                d2 = data_table[0, 1:] * y_scale
                d = data_table[1:, 1:] * z_scale

                # Check for increasing values (needed for interpolation class)
                if d1[1] < d1[0]:
                    d1 = numpy.flipud(d1)
                    d = numpy.flipud(d)
                if d2[1] < d2[0]:
                    d2 = numpy.flipud(d2)
                    d = numpy.fliplr(d)

                D1, D2 = numpy.meshgrid(d1, d2)

                intp_func = scipy.interpolate.RectBivariateSpline(d1, d2, d)

                out = intp_func, '2D', d, D1, D2, d1, d2
            else:  # 1D interpolation
                d1 = data_table[1:, 0] * x_scale
                d = data_table[1:, 1] * y_scale

                # Check for increasing values
                if d1[1] < d1[0]:
                    d1 = numpy.flipud(d1)
                    d = numpy.flipud(d)

                intp_func = scipy.interpolate.interp1d(d1, d)

                out = intp_func, '1D', d, d1
        else:
            print 'Data table format not correct!', '\n', 'Path:', fpath
            out = 'junk'

        return out

    def get_matl_properties(self, path):
        """
        Read in the material properties. Should be stored in
        data/matl_prop.txt, or something along those lines.
        """
        cfr = confreader.Reader(path)
        self.matl_prop = cfr.conf_data

        self.proc_matl_prop()

    def get_des_properties(self, path):
        """
        Read in the cell design properties. Should be stored in
        data/des_prop.txt, or something along those lines.
        """
        cfr = confreader.Reader(path)
        self.des_prop = cfr.conf_data

        self.proc_des_prop()

    def proc_matl_prop(self, ):
        """
        After running the get_matl_prop...() function, this is run to extract
        and compute the required information.
        """
        mp = self.matl_prop

        self.sig_a = mp['SOLID_CONDUCTIVITY']['sig_a']
        self.sig_c = mp['SOLID_CONDUCTIVITY']['sig_c']

        self.sc_a = mp['SITE_CAPACITY']['sc_a']
        self.sc_c = mp['SITE_CAPACITY']['sc_c']

        self.Mm_a = mp['MOLEC_MASS']['molec_mass_a']
        self.Mm_c = mp['MOLEC_MASS']['molec_mass_c']

        self.rho_act_a = mp['DENS']['activmatl_a']
        self.rho_act_c = mp['DENS']['activmatl_c']

        self.rho_binders = mp['DENS']['binders']
        self.rho_condaddtvs = mp['DENS']['cond_addtvs']

        self.num_act_a = len(self.rho_act_a)
        self.num_act_c = len(self.rho_act_c)
        self.num_binders = len(self.rho_binders)
        self.num_condaddtvs = len(self.rho_condaddtvs)

        self.rho_electrolyte = mp['DENS']['electrolyte']

        self.t_plus = mp['ELECTROLYTE']['transference_number']

        # Max concentration in anode,   [mol/m^3]
        self.csa_max = (self.sc_a * 3600 / self.F) * (self.rho_act_a[0])
        # Max concentration in cathode, [mol/m^3]
        self.csc_max = (self.sc_c * 3600 / self.F) * (self.rho_act_c[0])

    def proc_des_prop(self, ):
        """
        After running the get_matl_prop...(), proc_matl_prop...(), and
        get_des_prop...() functions, this is run to extract and compute the
        required cell design related information.
        """
        dp = self.des_prop

        # Particle radii
        self.Rp_a = dp['PARTICLE']['Rp_a']
        self.Rp_c = dp['PARTICLE']['Rp_c']

        # Bruggeman correction terms for tortuosity
        self.brug_a = dp['TORTUOSITY']['brug_a']
        self.brug_s = dp['TORTUOSITY']['brug_s']
        self.brug_c = dp['TORTUOSITY']['brug_c']

        # Geometry related items
        # Domain thicknesses
        self.La = dp['GEOMETRY']['La']
        self.Ls = dp['GEOMETRY']['Ls']
        self.Lc = dp['GEOMETRY']['Lc']

        # Coated area (for current density calculations)
        self.Area = dp['COATING']['HEIGHT'] * dp['COATING']['LENGTH']
        # Mass loading
#        self.an_mL  = dp['MASS_LOADING']['an_mL'][0]
#        self.cat_mL = dp['MASS_LOADING']['cat_mL'][0]
#        self.AM_frac_n = dp['MASS_LOADING']['an_AMs'][0]
#        self.AM_frac_p = dp['MASS_LOADING']['cat_AMs'][0]

        # Volume fractions, TODO: need to change for multichem
        self.epsilon_e_a = dp['VOL_FRACS']['epsilon_e_a']
        self.epsilon_e_s = dp['VOL_FRACS']['epsilon_e_s']
        self.epsilon_e_c = dp['VOL_FRACS']['epsilon_e_c']

        self.epsilon_f_a = dp['VOL_FRACS']['epsilon_f_a']
        self.epsilon_f_c = dp['VOL_FRACS']['epsilon_f_c']

        # Pulled over from dae_genPart_T.py testing code, needs cleaner
        # integration
        eps_a = self.epsilon_e_a
        eps_s = self.epsilon_e_s
        eps_c = self.epsilon_e_c
        # 1.3, 0.5, 0.5 #0.95, 0.5, 0.35
        ba, bs, bc = self.brug_a, self.brug_s, self.brug_c

        # list( eps_a + eps_a/2.*numpy.sin(numpy.linspace(0.,Na/4,Na)) ) #
        # list(eps_a + eps_a*numpy.random.randn(Na)/5.) #
        eps_a_vec = [eps_a for i in range(self.Na)]
        eps_s_vec = [eps_s for i in range(self.Ns)]
        # list( eps_c + eps_c/2.*numpy.sin(numpy.linspace(0.,Nc/4,Nc)) ) #
        # list(eps_c + eps_c*numpy.random.randn(Nc)/5.) #
        eps_c_vec = [eps_c for i in range(self.Nc)]

        self.eps_a_vec = eps_a_vec
        self.eps_s_vec = eps_s_vec
        self.eps_c_vec = eps_c_vec

        self.eps_m = numpy.array(eps_a_vec + eps_s_vec + eps_c_vec, dtype='d')
        self.k_m = 1. / self.eps_m
        self.K_m = numpy.diag(self.k_m)

        self.eps_mb = numpy.array([ea**ba for ea in eps_a_vec] +
                                  [es**bs for es in eps_s_vec] +
                                  [ec**bc for ec in eps_c_vec], dtype='d')
        # For effective parameters, e.g. ke_eff and De_eff
        self.eps_eff = numpy.array([ea**(1. + ba) for ea in eps_a_vec]
                                   + [es**(1. + bs) for es in eps_s_vec]
                                   + [ec**(1. + bc) for ec in eps_c_vec],
                                   dtype='d')

        self.eps_a_eff = self.eps_eff[:self.Na]
        self.eps_c_eff = self.eps_eff[-self.Nc:]

        self.eps_s_a_eff = 1. - self.eps_a_eff
        self.eps_s_c_eff = 1. - self.eps_c_eff

        # Solid phase volume fraction of active material
        self.eps_s_a = 1. - \
            numpy.array(eps_a_vec, dtype='d') - self.epsilon_f_a
        self.eps_s_c = 1. - \
            numpy.array(eps_c_vec, dtype='d') - self.epsilon_f_c

        as_a = 3. * (self.eps_s_a) / self.Rp_a
        as_c = 3. * (self.eps_s_c) / self.Rp_c
        self.as_a = as_a
        self.as_c = as_c

#        EASA_a = as_a[0] #1.70*(2.25*100**3) #*numpy.ones_like( eps_a_vec )
#        EASA_c = as_c[0] #0.26*(4.74*100**3) #*numpy.ones_like( eps_c_vec )

#        self.io_coeff_a = EASA_a/self.as_a
#        self.io_coeff_c = EASA_c/self.as_c

    def gen_thermFactor(self, x, U, dUdx, bern_on):
        """
        Compute the activity correction term from the equilibrium potential
        data.
        One of two methods may be used. Dawn Bernardi, et al. have a technique
        and essentially all other use a slightly different method. Both are
        compute and an input control parameter in the conf file may be used to
        switch between the two here.
        """
        if bern_on:
            tf = (0 + -(self.Faraday / self.R / 298.15 *
                        x * (1. - x)) * dUdx) / (1. - x)
            thermFactor = tf * (1. / abs(numpy.mean(tf)))
        else:
            tf = (0 + -(self.Faraday / self.R / 298.15 * x * (1. - x)) * dUdx)
            thermFactor = tf * (1. / abs(numpy.mean(tf)))

        return thermFactor

    def get_fullstoichrange_ocv(self):
        """
        Compute the full ocv curve data beyond the stoich limits provide for
        alignment purposes.
        This is needed as sometimes the initial stoichs given may be say at
        3.0V and 4.1V full cell voltages, however, one may actually be able
        to take the cell to 2.5V and 4.2V based all of the equilibrium curve
        data provided for each half cell.
        """
        # Determine new bottom cathode stoich
        # assume bottom of ocv is when anode is at 0.0 stoich
        # remaining Ah from bott cap anode value
        anbottcap = (self.sc_n * (self.Udat_n['botV_x'] - self.Udat_n['min_x'])
                     * (self.an_mL * self.AM_frac_n) * self.Area)  # [Ah]
        deltacat = (anbottcap /
                    (self.sc_p * (self.cat_mL * self.AM_frac_p) * self.Area))

        # Update bottom stoichs to push anode to 0.0
        self.cat_botV_x = self.Udat_p['botV_x'] + deltacat
        self.an_botV_x = self.Udat_n['min_x']

        min_cat_dod = (1. - self.Udat_p['botV_x']) * \
            (self.sc_p * (self.cat_mL * self.AM_frac_p) * self.Area)

        # Determine top stoichs such that U_full_top = 4.2V
        dod_an = self.sc_n * (self.an_mL * self.AM_frac_n) * \
            self.Area * self.Udat_n['x']
        dod_cat_rough = self.sc_p * \
            (self.cat_mL * self.AM_frac_p) * self.Area * self.Udat_p['x']
        dod_cat = dod_cat_rough.max() - dod_cat_rough - min_cat_dod

        max_dod = numpy.amin([dod_cat.max(), dod_an.max()]) - 0.001
        min_dod = numpy.amax([dod_cat.min(), dod_an.min()]) + 0.001

        dod = numpy.linspace(min_dod, max_dod, num=500, dtype='d')

        Up = numpy.interp(
            dod, numpy.flipud(
                dod_cat.flatten()), numpy.flipud(
                self.Udat_p['U'].flatten()))
        Ua = numpy.interp(
            dod, numpy.flipud(
                dod_an.flatten()), numpy.flipud(
                self.Udat_n['U'].flatten()))

        ocv = Up - Ua

        dod_42 = numpy.interp(4.2, ocv, dod)

        Up_42 = numpy.interp(
            dod_42, numpy.flipud(
                dod_cat.flatten()), numpy.flipud(
                self.Udat_p['U'].flatten()))
        Ua_42 = numpy.interp(
            dod_42, numpy.flipud(
                dod_an.flatten()), numpy.flipud(
                self.Udat_n['U'].flatten()))

        self.Ah_cap = dod_42
        self.Udat_p['topV_x'] = numpy.interp(
            Up_42, numpy.flipud(
                self.Udat_p['U'].flatten()), numpy.flipud(
                self.Udat_p['x'].flatten()))
        self.Udat_n['topV_x'] = numpy.interp(
            Ua_42, self.Udat_n['U'].flatten(), self.Udat_n['x'].flatten())

    def get_fullcell_ocv(self, V_init):
        """
        Setup the full cell ocv curve using the half cell potential data
        """
        nSoc = 200
        self.an_soc = numpy.linspace(
            self.Udat_n['topV_x'],
            self.Udat_n['botV_x'],
            num=nSoc,
            dtype='d')
        self.cat_soc = numpy.linspace(
            self.Udat_p['topV_x'],
            self.Udat_p['botV_x'],
            num=nSoc,
            dtype='d')

        an_pot = numpy.interp(
            self.an_soc, numpy.flipud(
                self.Udat_n['x']), numpy.flipud(
                self.Udat_n['U']))
        cat_pot = numpy.interp(
            self.cat_soc,
            self.Udat_p['x'],
            self.Udat_p['U'])

        self.nom_cap = 2.15 * (self.Area / 0.070452)

        self.xs = numpy.linspace(1., 0., nSoc, dtype='d')

        self.full_soc_an = numpy.interp(
            self.Udat_n['x'], numpy.flipud(
                self.an_soc), numpy.flipud(
                self.xs))
        self.full_soc_cat = numpy.interp(
            self.Udat_p['x'], self.cat_soc, self.xs)

        self.Up = cat_pot
        self.Ua = an_pot

        self.OCP = self.Up - self.Ua

        if self.RunInput['MODEL']['ELECTRODE'] == 'full' \
                or self.RunInput['MODEL']['ELECTRODE'] == 'anode':
            soc_init = numpy.interp(
                V_init, numpy.flipud(
                    self.OCP), numpy.flipud(
                    self.xs))

            Up_i = numpy.interp(
                soc_init, numpy.flipud(
                    self.xs), numpy.flipud(
                    self.Up))
            Ua_i = numpy.interp(
                soc_init, numpy.flipud(
                    self.xs), numpy.flipud(
                    self.Ua))

            self.V_init = Up_i - Ua_i

            self.theta_n0 = numpy.interp(Ua_i, an_pot, self.an_soc)
            self.theta_p0 = numpy.interp(
                Up_i, numpy.flipud(cat_pot), numpy.flipud(
                    self.cat_soc))

            self.OCP_sim = numpy.concatenate(
                [[V_init, ], self.OCP[self.OCP < V_init]])
            self.xs_sim = numpy.interp(
                self.OCP_sim, numpy.flipud(
                    self.OCP), numpy.flipud(
                    self.xs))
            self.ah_sim = (1. - self.xs_sim) * self.Ah_cap

        elif self.RunInput['MODEL']['ELECTRODE'] == 'cathode':
            cat_x_init = numpy.interp(
                V_init, numpy.flipud(
                    self.Udat_p['U']), numpy.flipud(
                    self.Udat_p['x']))
            self.theta_n0 = 0.5083

            Up_i = V_init
            Ua_i = numpy.interp(self.theta_n0, self.an_soc, self.Ua)

            self.theta_p0 = cat_x_init

            ocp_init = numpy.interp(Up_i, self.Up, self.OCP)

            self.OCP_sim = numpy.concatenate(
                [[ocp_init, ], self.OCP[self.OCP < ocp_init]])
            self.xs_sim = numpy.interp(self.OCP_sim, self.OCP, self.xs)
            self.ah_sim = (1. - self.xs_sim) * self.Ah_cap

    def get_init_thetas(self, V_init, an_x_top, cat_x_top):
        """
        From V_initial, use the stoich alignment and get the initial thetas for
        each electrode.
        """
        vol_a = numpy.mean(self.eps_s_a) * self.Area * self.La
        vol_c = numpy.mean(self.eps_s_c) * self.Area * self.Lc

        m_a = vol_a * self.rho_act_a[0]
        m_c = vol_c * self.rho_act_c[0]

        cap = self.RunInput['MODEL']['OCV_CAP']

        an_x_bott = an_x_top - cap / m_a / self.sc_a
        cat_x_bott = cat_x_top + cap / m_c / self.sc_c

#        cap_a = m_a * self.sc_a * ( an_x_top - 0.0 )
#        cap_c = m_c * self.sc_c * ( 1.0 - cat_x_top )

#        print cap_a
#        print cap_c

#        if cap_a <= cap_c : # anode limited at bottom of discharge (low-rates)
#            an_x_bott  = 0.0
#            cat_x_bott = cat_x_top + cap_a/(m_c*self.sc_c)
#        else : # cathode limited at bottom of discharge (low-rate)
#            cat_x_bott = 1.0
#            an_x_bott  = an_x_top  - cap_c/(m_a*self.sc_a)

        NX = 200
        axi = numpy.flipud(numpy.linspace(an_x_top, an_x_bott, NX))
        cxi = numpy.linspace(cat_x_top, cat_x_bott, NX)

        U_a = numpy.flipud(self.uref_a(axi))
        U_c = self.uref_c(cxi)

        ax = numpy.flipud(axi)

        OCV = U_c - U_a

        x = numpy.linspace(0., 1., NX)

        self.xs = 1.0 - x  # convert dod to soc
        self.OCP = OCV

        x_init = numpy.interp(V_init, numpy.flipud(OCV), numpy.flipud(x))

        Ua_init = numpy.interp(x_init, x, U_a)
        Uc_init = numpy.interp(x_init, x, U_c)

        self.theta_a0 = numpy.interp(Ua_init, U_a, ax)
        self.theta_c0 = numpy.interp(
            Uc_init, numpy.flipud(U_c), numpy.flipud(cxi))
