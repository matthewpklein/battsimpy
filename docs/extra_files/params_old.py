import numpy
#import csv
import confreader
import scipy.interpolate
import batteqns 

from matplotlib import pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D


class params( ) :

    def __init__( self ) :
        """
        Initialize the class
        """
        self.pi = 3.14159265358

    def buildpars( self, V_init, Pdat ) :
        """
        Parameter builder for all models in BattSimPy
        """
        self.V_init = V_init
        self.Pdat = Pdat

        self.RunInput = Pdat['RunInput']
        RunInput      = Pdat['RunInput']
        fname_root    = RunInput['FILEPATHS']['DATA_ROOT'] + \
                        'Model_v' + RunInput['FILEPATHS']['MODEL_NUM'] + '/' + \
                        RunInput['FILEPATHS']['PARAMS'] + '/'

        # Check if physics type model
        if 'ecm' in RunInput['MODEL']['MODEL_TYPE'] or 'ECM' in RunInput['MODEL']['MODEL_TYPE'] :
            self.modelPhysical = 0
        else :
            self.modelPhysical = 1

        # Universal constants
        self.Faraday = 96487.0   # [Coulombs/mol]
        self.R       = 8.314472 # [J/(mol-K)]

        # T used internally here for some initilization of parameters
        T_init = 298.15 # [K]

        ## Parameters unique to physics based models
        if self.modelPhysical :
            # Mesh sizes
            Nx  = int( RunInput['MESH']['NX']  ) # Number of points in the x dimension
            NRN = int( RunInput['MESH']['NRN'] ) # Number of points in the negative particle dim
            NRP = int( RunInput['MESH']['NRP'] ) # Number of points in the positive particle dim
            
            # io optimization boolean control variables
            self.kn_opt_on = RunInput['OPTIMIZATION']['ION_OPT_ON']
            self.kp_opt_on = RunInput['OPTIMIZATION']['IOP_OPT_ON']

            # Particle discretization variables
            self.cstype = RunInput['MESH']['CS_TYPE']
            WN = RunInput['MESH']['CS_WN']
            WP = RunInput['MESH']['CS_WP']

            # Solid phase transport (Ds) parameters
            self.Dsdat_p = { 'stoich_sens_on':RunInput['SOLID_DIFFUSION']['VAR_DIFF_CATHODE_ON'] }
            self.Dsdat_n = { 'stoich_sens_on':RunInput['SOLID_DIFFUSION']['VAR_DIFF_ANODE_ON'] }

             # Radial Ds sensitivity control (e.g., If Ds is lower at particle surface) 
            self.Dsdat_n['r_sens_on'] = RunInput['SOLID_DIFFUSION']['R_DIFF_ANODE_ON']
            self.Dsdat_p['r_sens_on'] = RunInput['SOLID_DIFFUSION']['R_DIFF_CATHODE_ON']
            if self.Dsdat_p['r_sens_on'] :
                self.Dsdat_p['DsR'] = {'r':[], 'coeff':[]}
                DsR_cat = numpy.loadtxt( fname_root+'solid/diffusion/'+'DsR_cathode.csv', dtype='d', delimiter=',' )
                self.Dsdat_p['r'] = DsR_cat[:,0].flatten()
                self.Dsdat_p['coeff'] = DsR_cat[:,1].flatten()

            if self.Dsdat_n['r_sens_on'] :
                self.Dsdat_n['DsR'] = {'r':[], 'coeff':[]}
                DsR_an = numpy.loadtxt( fname_root+'solid/diffusion/'+'DsR_anode.csv', dtype='d', delimiter=',' )
                self.Dsdat_n['r'] = DsR_an[:,0].flatten()
                self.Dsdat_n['coeff'] = DsR_an[:,1].flatten()

            # Material properties
            self.get_matl_properties( fname_root+'matl_prop.txt' )
            self.c_e = self.matl_prop['ELECTROLYTE']['c_e_init']

            # Design properties
            self.get_des_properties( fname_root+'des_prop.txt' )

            # Discretization through sandwich thickness
            L  = self.L_n+self.L_s+self.L_p # [m]
            dx = L/(Nx-1) # [m]

            Nn = int( round( self.L_n/dx ) )
            Np = int( round( self.L_p/dx ) )
            Ns = int( Nx - Nn - Np )

            self.dxn = self.L_n/(Nn-1)   # [m]
            self.dxs = self.L_s/(Ns+2-1) # [m]
            self.dxp = self.L_p/(Np-1)   # [m]

            # Tab and foil ohmic resistances
            self.Rfl = RunInput['MODEL']['FOIL_RES'] # [Ohms]
            self.Rtb = RunInput['MODEL']['TAB_RES']  # [Ohms]
            # Cathode ohmic resistance used for single particle model
            self.Rocat = 0.0 # [Ohms]

            # Volume fractions
            self.epsilon_s_n = 1. - self.epsilon_e_n - self.epsilon_f_n
            self.epsilon_s_p = 1. - self.epsilon_e_p - self.epsilon_f_p

            # Total active electrochemical area
            self.a_s_n = 3. * self.epsilon_s_n / self.R_s_n
            self.a_s_p = 3. * self.epsilon_s_p / self.R_s_p

            self.S_n = self.a_s_n * self.L_n * self.Area
            self.S_p = self.a_s_n * self.L_p * self.Area

            # PSD stuff (not yet)

            # Active material electronic conductivity
            self.sig_n_eff = self.matl_prop['SOLID_CONDUCTIVITY']['sig_n'] * (1.-self.epsilon_e_n)**(1.+self.brug_n) # [1/Ohm-m]
            self.sig_p_eff = self.matl_prop['SOLID_CONDUCTIVITY']['sig_p'] * (1.-self.epsilon_e_p)**(1.+self.brug_p) # [1/Ohm-m]

            # Electrolyte transport properties
            self.activ_on = RunInput['ELECTROLYTE']['ACTIVITY_ON']
            De_fn    = fname_root + 'electrolyte/' + RunInput['ELECTROLYTE']['DE_FN'] 
            kappa_fn = fname_root + 'electrolyte/' + RunInput['ELECTROLYTE']['KAP_FN'] 
            fca_fn   = fname_root + 'electrolyte/' + RunInput['ELECTROLYTE']['FCA_FN']

             # Electrolyte salt diffusivity
            De_fac = RunInput['ELECTROLYTE']['DE_FACTOR']
            self.De_intp_table, De_tab_dim, self.De_d, self.De_ce, self.De_T, d1, d2 = self.gen_param_interp_function( De_fn, z_scale=De_fac )
#            Dedat = numpy.loadtxt( De_fn, dtype='d', delimiter=',' )
#            d1 = Dedat[1:,0]
#            d2 = Dedat[0,1:]
#            self.De_d = Dedat[1:,1:]*De_fac
#            self.De_ce, self.De_T = numpy.meshgrid( d1, d2 )
#            self.De_intp_table    = scipy.interpolate.RectBivariateSpline( d1, d2, self.De_d )

            [grad_De_d , junk] = numpy.gradient(self.De_d)
            [grad_De_ce, junk] = numpy.gradient(self.De_ce.T)
            grad_De_dce = grad_De_d / grad_De_ce
            self.dDe_dce_intp_table = scipy.interpolate.RectBivariateSpline( d1, d2, grad_De_dce )

             # Electrolyte ionic conductivity
            ke_fac = RunInput['ELECTROLYTE']['KE_FACTOR']
            self.kap_intp_table, kap_tab_dim, self.kap_d, self.kap_ce, self.kap_T, d1, d2 = self.gen_param_interp_function( kappa_fn, z_scale=ke_fac )
##            kedat  = numpy.loadtxt( kappa_fn, delimiter=',' )
##            d1 = kedat[1:,0]
##            d2 = kedat[0,1:]
##            self.kap_d = 0.1*kedat[1:,1:]*ke_fac  # 0.1 converts from the mS/cm data to S/m units
##            self.kap_ce, self.kap_T = numpy.meshgrid( d1, d2 )
##            self.kap_intp_table = scipy.interpolate.RectBivariateSpline( d1, d2, self.kap_d )

            [grad_kap_d , junk] = numpy.gradient(self.kap_d)
            [grad_kap_ce, junk] = numpy.gradient(self.kap_ce.T)
            grad_kap_dce = grad_kap_d / grad_kap_ce
            self.dkap_dce_intp_table = scipy.interpolate.RectBivariateSpline( d1, d2, grad_kap_dce )

            self.R_ke  = RunInput['ELECTROLYTE']['RKE']
            self.Ea_ke = RunInput['ELECTROLYTE']['EA_KE']

             # Electrolyte activity correction
            self.fca_intp_table, fca_tab_dim, self.fca_d, self.fca_ce, self.fca_T, d1, d2 = self.gen_param_interp_function( fca_fn )
#            fcadat = numpy.loadtxt( fca_fn, dtype='d', delimiter=',' )
#            d1 = fcadat[1:,0]
#            d2 = fcadat[0,1:]
#            self.fca_d = fcadat[1:,1:]
#            self.fca_ce, self.fca_T = numpy.meshgrid( d1, d2 )
#            self.fca_intp_table = scipy.interpolate.RectBivariateSpline( d1, d2, self.fca_d )

            [grad_fca_d , junk] = numpy.gradient(self.fca_d)
            [grad_fca_ce, junk] = numpy.gradient(self.fca_ce.T)
            grad_fca_dce = grad_fca_d / grad_fca_ce
            self.dfca_dce_intp_table = scipy.interpolate.RectBivariateSpline( d1, d2, grad_fca_dce )

            ## --- Kinetic paramters --- ##
            self.kn_fn = fname_root+'solid/kinetics/' + RunInput['KINETICS']['ION_FN']
            self.kp_fn = fname_root+'solid/kinetics/' + RunInput['KINETICS']['IOP_FN']

            self.kn_coeff = RunInput['KINETICS']['IOP_COEF']
            self.kp_coeff = RunInput['KINETICS']['ION_COEF']

            self.ion_dat = { 'io_interp_on':RunInput['KINETICS']['ION_INTERP_ON'], 'alpha_n':0.5, 'alpha_p':0.5, 'csmax':self.c_s_n_max }
            self.iop_dat = { 'io_interp_on':RunInput['KINETICS']['IOP_INTERP_ON'], 'alpha_n':0.5, 'alpha_p':0.5, 'csmax':self.c_s_p_max }
            
            # Anode io parameter map loading
            if self.ion_dat['io_interp_on'] :
                self.ion_dat['io_map'], ion_tab_dim, d, D1, D2, d1, d2 = self.gen_param_interp_function( self.kn_fn )
#                kndat = numpy.loadtxt( self.kn_fn, dtype='d', delimiter=',' )
#                d1 = numpy.flipud( kndat[1:,0] )
#                d2 = kndat[0,1:]
#                d  = numpy.flipud( kndat[1:,1:] )
#                D1, D2 = numpy.meshgrid( d1, d2 )
#                self.ion_dat['io_map'] = scipy.interpolate.RectBivariateSpline( d1, d2, d ) #interp2d( D1, D2, d )

                [grad_ion_d , junk] = numpy.gradient(d)
                [grad_ion_xs, junk] = numpy.gradient(D1.T)
                grad_dion_dcs = grad_ion_d / (grad_ion_xs*self.c_s_n_max)
                self.ion_dat['dio_dcs_map'] = scipy.interpolate.RectBivariateSpline( d1, d2, grad_dion_dcs )

                [junk, grad_ion_d] = numpy.gradient(d)
                [junk, grad_ion_T] = numpy.gradient(D2.T)
                grad_dion_dT = grad_ion_d / grad_ion_T
                self.ion_dat['dio_dT_map'] = scipy.interpolate.RectBivariateSpline( d1, d2, grad_dion_dT )
            else :
                self.ion_dat['D_io'] = 5.0
                self.ion_dat['k_Ea'] = 0.0
                # TODO: deal with constant input io values

            # Cathode parameter map loading
            if self.iop_dat['io_interp_on'] :
                self.iop_dat['io_map'], iop_tab_dim, d, D1, D2, d1, d2 = self.gen_param_interp_function( self.kp_fn )
#                kpdat = numpy.loadtxt( self.kp_fn, dtype='d', delimiter=',' )
#                d1 = kpdat[1:,0]
#                d2 = kpdat[0,1:]
#                d = kpdat[1:,1:]
#                D1, D2 = numpy.meshgrid( d1, d2 )
#                self.iop_dat['io_map'] = scipy.interpolate.RectBivariateSpline( d1, d2, d )

                [grad_iop_d , junk] = numpy.gradient(d)
                [grad_iop_xs, junk] = numpy.gradient(D1.T)
                grad_diop_dcs = grad_iop_d / (grad_iop_xs*self.c_s_p_max)
                self.iop_dat['dio_dcs_map'] = scipy.interpolate.RectBivariateSpline( d1, d2, grad_diop_dcs )
                
                [junk, grad_iop_d] = numpy.gradient(d)
                [junk, grad_iop_T] = numpy.gradient(D2.T)
                grad_diop_dT = grad_iop_d / grad_iop_T
                self.iop_dat['dio_dT_map'] = scipy.interpolate.RectBivariateSpline( d1, d2, grad_diop_dT )
            else :
                self.iop_dat['D_io'] = 5.0
                self.iop_dat['k_Ea'] = 0.0
                # TODO: deal with constant input io values

            # Film resistance parameter loading (TODO: determine the film_res file format and load method)
            #Rf_dat = conf_reader.reader( fname_root+'/'+RunInput['film_res.csv'] )
            self.R_f_n  = 0. #Rf_dat['R_f_n']   # [Ohms-m^2]
            self.R_f_p  = 0. #Rf_dat['R_f_p']   # [Ohms-m^2]
            self.Rfn_Ea = 0. #Rf_dat['Rfn_Ea']  # [Ohms-m^2]
            self.Rfp_Ea = 0. #Rf_dat['Rfp_Ea']  # [Ohms-m^2]

            # Double layer capacitance (TODO: add SPM_dl)
            #self.Cdl_n = 0.2
            #self.Cdl_p = 1.2
            #self.Cdl_kmm = 1.2

            # Electrolyte reference concentration
            self.c_e_ref = 1000. # [mol/m^3]

            # Particle discretization
            self.SolidOrder_n = NRN
            self.SolidOrder_p = NRP

            if self.cstype == 'nonunif' :
                self.rn = batteqns.nonlinspace( self.R_s_n, WN, self.SolidOrder_n+1 ) # FVM
                self.rp = batteqns.nonlinspace( self.R_s_p, WP, self.SolidOrder_p+1 )

                self.drn = numpy.gradient( self.rn )
                self.drp = numpy.gradient( self.rp )

            else :
                self.drn = self.R_s_n/(self.SolidOrder_n)
                self.drp = self.R_s_p/(self.SolidOrder_p)
                self.rn = numpy.array([range(self.SolidOrder_n+1)], dtype='d')*self.drn
                self.rp = numpy.array([range(self.SolidOrder_p+1)], dtype='d')*self.drp

            self.rn_m = numpy.array( [ (self.rn[ir+1]+self.rn[ir])/2.0 for ir in range(len(self.rn)-1) ] )
            self.rp_m = numpy.array( [ (self.rp[ir+1]+self.rp[ir])/2.0 for ir in range(len(self.rp)-1) ] )

            self.v_n = numpy.array( [ 1./3.*((self.rn[i+1]**3.) - (self.rn[i]**3.)) for i in range(len(self.rn)-1) ] )
            self.v_p = numpy.array( [ 1./3.*((self.rp[i+1]**3.) - (self.rp[i]**3.)) for i in range(len(self.rp)-1) ] )

            self.A1n, self.A2n, self.Bn = batteqns.pre_csmat_fvm( self.rn, self.rn_m, self.v_n )
            self.A1p, self.A2p, self.Bp = batteqns.pre_csmat_fvm( self.rp, self.rp_m, self.v_p )

            self.rn_intp = numpy.concatenate( [ [self.rn[0]], self.rn_m, [self.R_s_n,] ] )
            self.rp_intp = numpy.concatenate( [ [self.rp[0]], self.rp_m, [self.R_s_p,] ] )

            self.csdat_n = batteqns.particle_BCs_fvm(self.rn,self.rn_m)
            self.csdat_p = batteqns.particle_BCs_fvm(self.rp,self.rp_m)

            self.csdat_n['C_cs_bar'] = batteqns.particle_vols_fvm(self.rn)
            self.csdat_p['C_cs_bar'] = batteqns.particle_vols_fvm(self.rp)

            # Additional csdat params added for the solid_diffusion function in full_1d
            self.csdat_n['A11'] = self.A1n
            self.csdat_n['A22'] = self.A2n
            self.csdat_n['B1']  = self.Bn
            
            self.csdat_p['A11'] = self.A1p
            self.csdat_p['A22'] = self.A2p
            self.csdat_p['B1']  = self.Bp
            
            self.csdat_p['r'] = self.rp
            self.csdat_n['r'] = self.rn

            self.csdat_p['r_intp'] = self.rp_intp
            self.csdat_n['r_intp'] = self.rn_intp

            self.csdat_p['r_m'] = self.rp_m
            self.csdat_n['r_m'] = self.rn_m

            self.Nn_x = Nn
            self.Np_x = Np
            
            self.Nn = Nn-2
            self.Ns = Ns
            self.Np = Np-2

            self.Nn_cs = self.Nn
            self.Np_cs = self.Np

            self.csn_inds = range(self.Nn)
            self.csp_inds = range(self.Np)

            self.Nce = self.Nn+self.Ns+self.Np
            self.Nx  = Nx

            self.xF = list(numpy.linspace(0,self.L_n,Nn)) + list(numpy.linspace(self.L_n+self.dxs,self.L_n+self.L_s,Ns+2)) + list(numpy.linspace(self.L_n+self.L_s+self.dxp,self.L_n+self.L_s+self.L_p,Np))

            self.xf = list(numpy.linspace(self.dxn,self.L_n-self.dxn,self.Nn)) + list(numpy.linspace(self.L_n+self.dxs,self.L_n+self.L_s-self.dxs,self.Ns)) + list(numpy.linspace(self.L_n+self.L_s+self.dxp,self.L_n+self.L_s+self.L_p-self.dxp,self.Np))

            self.Ncsn = (self.SolidOrder_n)*self.Nn_cs
            self.Ncsp = (self.SolidOrder_p)*self.Np_cs

            self.Nnp_cs = self.Nn_cs + self.Np_cs

            self.Nc    = self.Ncsn+self.Ncsp+self.Nce
            self.Nxvec = self.Nc+1
            self.Nnp   = self.Nn+self.Np
            self.Nz    = 3*self.Nnp + self.Nce

            # x inds
            self.ind_csn = range(self.Ncsn)
            self.ind_csn_r = numpy.reshape( self.ind_csn, [len(self.ind_csn),1] )
            self.ind_csn_c = numpy.reshape( self.ind_csn, [1,len(self.ind_csn)] )
            self.ind_csp = range(self.Ncsn,self.Ncsn+self.Ncsp)
            self.ind_csp_r = numpy.reshape( self.ind_csp, [len(self.ind_csp),1] )
            self.ind_csp_c = numpy.reshape( self.ind_csp, [1,len(self.ind_csp)] )

            self.ind_cs  = range(self.Ncsn+self.Ncsp)
            self.ind_ce  = range(self.Ncsn+self.Ncsp,self.Nc)
            self.ind_ce_r = numpy.reshape( self.ind_ce, [len(self.ind_ce),1] )
            self.ind_ce_c = numpy.reshape( self.ind_ce, [1,len(self.ind_ce)] )
            
            self.ind_T = self.ind_ce[-1]+1

            # z inds
            self.ind_phisn = range(self.Nn)
            self.ind_phisn_r = numpy.reshape( self.ind_phisn, [len(self.ind_phisn),1] )
            self.ind_phisn_c = numpy.reshape( self.ind_phisn, [1,len(self.ind_phisn)] )
            
            self.ind_phisp = range(self.Nn, self.Nnp)
            self.ind_phisp_r = numpy.reshape( self.ind_phisp, [len(self.ind_phisp),1] )
            self.ind_phisp_c = numpy.reshape( self.ind_phisp, [1,len(self.ind_phisp)] )

            self.ind_ien = range(self.Nnp, self.Nnp+self.Nn)
            self.ind_ien_r = numpy.reshape( self.ind_ien, [len(self.ind_ien),1] )
            self.ind_ien_c = numpy.reshape( self.ind_ien, [1,len(self.ind_ien)] )

            self.ind_iep = range(self.Nnp+self.Nn, 2*self.Nnp)
            self.ind_iep_r = numpy.reshape( self.ind_iep, [len(self.ind_iep),1] )
            self.ind_iep_c = numpy.reshape( self.ind_iep, [1,len(self.ind_iep)] )

            self.ind_ienp = self.ind_ien+self.ind_iep
            self.ind_ienp_c = numpy.reshape( self.ind_ienp, [1,len(self.ind_ienp)] )

            self.ind_phie = range(2*self.Nnp, 2*self.Nnp+self.Nce)
            self.ind_phie_r = numpy.reshape( self.ind_phie, [len(self.ind_phie),1] )
            self.ind_phie_c = numpy.reshape( self.ind_phie, [1,len(self.ind_phie)] )

            self.ind_jn = range(2*self.Nnp+self.Nce, 2*self.Nnp+self.Nce+self.Nn)
            self.ind_jn_r = numpy.reshape( self.ind_jn, [len(self.ind_jn),1] )
            self.ind_jn_c = numpy.reshape( self.ind_jn, [1,len(self.ind_jn)] )

            self.ind_jp = range(2*self.Nnp+self.Nce+self.Nn, self.Nz)
            self.ind_jp_r = numpy.reshape( self.ind_jp, [len(self.ind_jp),1] )
            self.ind_jp_c = numpy.reshape( self.ind_jp, [1,len(self.ind_jp)] )
            
            self.rsn = [ 1 for i in range(self.Nn) ]
            self.rsp = [ 1 for i in range(self.Np) ]
            self.csn = self.SolidOrder_n * numpy.ones(self.Nn_cs, dtype='int32')
            self.csp = self.SolidOrder_p * numpy.ones(self.Np_cs, dtype='int32')

    #        ## --- Gradient matrix building --- ##
            self.sec_ord   = 1

            self.delta_x_n = self.dxn/self.L_n
            self.delta_x_s = self.dxs/self.L_s
            self.delta_x_p = self.dxp/self.L_p

            g_n = batteqns.grad_secord( self.dxn, self.Nn+2, 0 )
            g_s = batteqns.grad_secord( self.dxs, self.Ns+2, 0 )
            g_p = batteqns.grad_secord( self.dxp, self.Np+2, 0 )

            self.gn = g_n
            self.gs = g_s
            self.gp = g_p

            gg_n = batteqns.gradgrad_secord( self.dxn, self.Nn+2, 0 )
            gg_s = batteqns.gradgrad_secord( self.dxs, self.Ns+2, 0 )
            gg_p = batteqns.gradgrad_secord( self.dxp, self.Np+2, 0 )

            self.ggn = gg_n
            self.ggs = gg_s
            self.ggp = gg_p

            # Equilibrium potentials
            self.Udat_n = {'temp_on':0} # intialize the Udat dict and turn off U temp sensitivity (TODO: create functionality for Temp sensitive U)
            self.Udat_p = {'temp_on':0}

            self.Udat_n['uref_fname'] = fname_root + 'solid/thermodynamics/' + RunInput['THERMODYNAMIC']['AN_POTENTIAL_FN']
            self.Udat_p['uref_fname'] = fname_root + 'solid/thermodynamics/' + RunInput['THERMODYNAMIC']['CAT_POTENTIAL_FN']

            # Stoich alignment parameters
            self.Udat_n['botV_x'] = 0.0
            self.Udat_n['topV_x'] = 0.8655

            self.Udat_p['botV_x'] = 0.95444
            self.Udat_p['topV_x'] = 0.34484

            # Load anode equil pot data
            an_num = numpy.loadtxt( self.Udat_n['uref_fname'], dtype='d', delimiter=',' )
            self.Udat_n['U'] = numpy.array( an_num[:,1], dtype='d' )
            self.Udat_n['x'] = numpy.array( 1.-an_num[:,0]/self.sc_n, dtype='d' )
            self.Udat_n['dUdx'] = numpy.gradient( self.Udat_n['U'] ) / numpy.gradient( self.Udat_n['x'] )

            self.Udat_n['max_x'] = self.Udat_n['x'].max()
            self.Udat_n['min_x'] = self.Udat_n['x'].min()
            self.Udat_n['max_U']   = self.Udat_n['U'].max()
            self.Udat_n['min_U']   = self.Udat_n['U'].min()

            # Load cathode equil pot data
            cat_num = numpy.loadtxt( self.Udat_p['uref_fname'], dtype='d', delimiter=',' )
            self.Udat_p['U'] = numpy.array( cat_num[:,1], dtype='d' )
            self.Udat_p['x'] = numpy.array( 1.-cat_num[:,0]/self.sc_p, dtype='d' )
            self.Udat_p['dUdx'] = numpy.gradient( self.Udat_p['U'] ) / numpy.gradient( self.Udat_p['x'] )

            self.Udat_p['max_x'] = self.Udat_p['x'].max()
            self.Udat_p['min_x'] = self.Udat_p['x'].min()
            self.Udat_p['max_U']   = self.Udat_p['U'].max()
            self.Udat_p['min_U']   = self.Udat_p['U'].min()

            # Compute activity correction coefficient for anode and cathode
            self.Udat_n['activity'] = self.gen_thermFactor( self.Udat_n['x'], self.Udat_n['U'], self.Udat_n['dUdx'], RunInput['SOLID_DIFFUSION']['BERNARDI_AN_ON'] )
            self.Udat_p['activity'] = self.gen_thermFactor( self.Udat_p['x'], self.Udat_p['U'], self.Udat_p['dUdx'], RunInput['SOLID_DIFFUSION']['BERNARDI_CAT_ON'] )

            # Shift the stoich alignment parameters to the maximum stoich range allowed
            # by the provided equilibrium data
            self.get_fullstoichrange_ocv()

            # theoretical maximum storage capacity using anode limits
            self.Ah_cap     = self.sc_n*abs(self.Udat_n['topV_x'] -self.Udat_n['botV_x'] )*(self.an_mL  * self.AM_frac_n)*self.Area
            # theoretical maximum storage capacity using cathode limits
            self.Ah_cap_cat = self.sc_p*abs(self.Udat_p['topV_x'] -self.Udat_p['botV_x'])*(self.cat_mL * self.AM_frac_p)*self.Area
            # Total moles of lithium in solid phase [mol/m^2]
            self.n_Li_s     = self.Ah_cap/self.Area*3600./self.Faraday

            # Compute the full cell OCV curve from the half cell curves and the stoich
            # alignment parameters
            self.get_fullcell_ocv( V_init )

            ## --- Ds, solid mass transport --- ##
             # Cathode
            if self.Dsdat_p['stoich_sens_on'] :
                Dsdat_cat = numpy.loadtxt( fname_root+'solid/diffusion/'+RunInput['SOLID_DIFFUSION']['DSP_FN'], dtype='d', delimiter=',' )
                datU = Dsdat_cat[(numpy.where(Dsdat_cat[:,0]>=self.Udat_p['min_U']) and numpy.where(Dsdat_cat[:,0]<=self.Udat_p['max_U'])),0].flatten()
                stoich = numpy.flipud(numpy.interp( numpy.flipud(datU.flatten()), numpy.flipud(self.Udat_p['U']), numpy.flipud(self.Udat_p['x']) ))
                Ds_cat = numpy.array( [ stoich, Dsdat_cat[:,1] ], dtype='d' ).T

                self.Dsdat_p['Ds_coeff']  = 1.0
                self.Dsdat_p['x']        = Ds_cat[:,0].flatten()
                self.Dsdat_p['Ds']       = Ds_cat[:,1].flatten()
                self.Dsdat_p['activity'] = numpy.interp( self.Dsdat_p['x'], self.Udat_p['x'], self.Udat_p['activity'] )

            else :
                self.Dsdat_p['Ds_coeff']  = RunInput['SOLID_DIFFUSION']['Dsp']
                self.Dsdat_p['x']        = numpy.linspace( self.Udat_p['min_x'], self.Udat_p['max_x'], 10, dtype='d' )
                self.Dsdat_p['Ds']       = numpy.ones(10, dtype='d') * self.Dsdat_p['Ds_coeff']
                self.Dsdat_p['activity'] = numpy.interp( self.Dsdat_p['x'], self.Udat_p['x'], self.Udat_p['activity'] )

            self.Dsdat_p['csmax']    = self.c_s_p_max
            self.Dsdat_p['therm_on'] = RunInput['SOLID_DIFFUSION']['TEMP_CAT_ON']

            self.Dsdat_p['const_Ea'] = RunInput['SOLID_DIFFUSION']['CONST_DSP_EA']
            if self.Dsdat_p['const_Ea'] :
                self.Dsdat_p['Ea']       = RunInput['SOLID_DIFFUSION']['Dsp_Ea']
            else :
                Dsp_Ea = numpy.loadtxt( fname_root+'solid/diffusion/'+RunInput['SOLID_DIFFUSION']['DSP_EA_FN'], dtype='d', delimiter=',' )
                datU = Dsp_Ea[(numpy.where(Dsp_Ea[:,0]>=self.Udat_p['min_U']) and numpy.where(Dsp_Ea[:,0]<=self.Udat_p['max_U'])),0].flatten()
                stoich = numpy.flipud(numpy.interp( numpy.flipud(datU.flatten()), numpy.flipud(self.Udat_p['U']), numpy.flipud(self.Udat_p['x']) ))
                self.Dsdat_p['Ea'] = numpy.interp( self.Dsdat_p['x'], stoich, Dsp_Ea[:,1].flatten() )

            self.Dsdat_p['activity_on'] = RunInput['SOLID_DIFFUSION']['ACTIVITY_CAT_ON']

            self.Dsdat_p['profile_on'] = 0 # used to be for setting a normalized Ds profile (TODO: use again or remove?)

             # Anode
            if self.Dsdat_n['stoich_sens_on'] :
                Dsdat_an = numpy.loadtxt( fname_root+'solid/diffusion/'+RunInput['SOLID_DIFFUSION']['DSN_FN'], dtype='d', delimiter=',' )
                datU = Dsdat_an[(numpy.where(Dsdat_an[:,0]>self.Udat_n['min_U']) and numpy.where(Dsdat_an[:,0]<self.Udat_n['max_U'])),0]
                stoich = numpy.interp( datU, self.Udat_n['U'], self.Udat_n['x'] ).flatten()
                Ds_an = numpy.array( [ stoich, Dsdat_an[:,1] ], dtype='d' ).T

                self.Dsdat_n['Ds_coeff']  = 1.0
                self.Dsdat_n['x']        = Ds_an[:,0].flatten()
                self.Dsdat_n['Ds']       = Ds_an[:,1].flatten()
                self.Dsdat_n['activity'] = numpy.interp( self.Dsdat_n['x'], self.Udat_n['x'], self.Udat_n['activity'] )

            else :
                self.Dsdat_n['Ds_coeff']  = RunInput['SOLID_DIFFUSION']['Dsn']
                self.Dsdat_n['x']        = numpy.linspace( 0.001, 0.999, 10, dtype='d' )
                self.Dsdat_n['Ds']       = numpy.ones(10, dtype='d') * self.Dsdat_n['Ds_coeff']
                self.Dsdat_n['activity'] = numpy.interp( self.Dsdat_n['x'], self.Udat_n['x'], self.Udat_n['activity'] )

            self.Dsdat_n['csmax']    = self.c_s_n_max
            self.Dsdat_n['therm_on'] = RunInput['SOLID_DIFFUSION']['TEMP_AN_ON']

            self.Dsdat_n['const_Ea'] = RunInput['SOLID_DIFFUSION']['CONST_DSN_EA']
            if self.Dsdat_n['const_Ea'] :
                self.Dsdat_n['Ea']       = RunInput['SOLID_DIFFUSION']['Dsn_Ea']
            else :
                Dsn_Ea = numpy.loadtxt( fname_root+'solid/diffusion/'+RunInput['SOLID_DIFFUSION']['DSN_EA_FN'], dtype='d', delimiter=',' )
                datU = Dsn_Ea[(numpy.where(Dsn_Ea[:,0]>=self.Udat_n['min_U']) and numpy.where(Dsn_Ea[:,0]<=self.Udat_n['max_U'])),0].flatten()
                stoich = numpy.flipud(numpy.interp( numpy.flipud(datU.flatten()), numpy.flipud(self.Udat_n['U']), numpy.flipud(self.Udat_n['x']) ))
                self.Dsdat_n['Ea'] = numpy.interp( self.Dsdat_n['x'], stoich, Dsn_Ea[:,1].flatten() )

            self.Dsdat_n['activity_on'] = RunInput['SOLID_DIFFUSION']['ACTIVITY_AN_ON']

            self.Dsdat_n['profile_on'] = 0 # used to be for setting a normalized Ds profile (TODO: use again or remove?)

            ## --- io optimization control parameters --- ##
            if self.kp_opt_on :
                self.k_dir  = self.kp_fn[:-len('io_cathode.csv')]
                self.kp_csv = numpy.loadtxt( self.k_dir+'kp_csv'+str(kopt_ind)+'.csv', dtype='d', delimiter=',' ) 
                kp_cp = self.kp_csv[:,0] * self.c_s_p_max
                self.kp_gradc = numpy.gradient( self.kp_csv[:,1] ) / numpy.gradient( kp_cp )

            if self.kn_opt_on :
                self.k_dir  = self.kn_fn[:-len('io_anode.csv')]
                self.kn_csv = numpy.loadtxt( self.k_dir+'kn_csv'+str(kopt_ind)+'.csv', dtype='d', delimiter=',' ) 
                kn_cn = self.kn_csv[:,0] * self.c_s_n_max
                self.kn_gradc = numpy.gradient( self.kn_csv[:,1] ) / numpy.gradient( kn_cn )

        ## Otherwise this is an empirical model (Equivalent Cicruit Model (ecm) type)
        else : 
            self.ecmOrder = RunInput['ECM_PARAMS']['ECM_ORDER']
            self.ecm_params = { 'ocv'    : {'intp_func':[],'dim':''}, 
                                'res_ohm': {'intp_func':[],'dim':''},
                                'res'    :[{'intp_func':[],'dim':''} for i_rc in range(self.ecmOrder)], 
                                'tau'    :[{'intp_func':[],'dim':''} for i_rc in range(self.ecmOrder)], }
            # OCV data
            self.ocv_fn = fname_root + 'ecm/ocv/' + self.RunInput['ECM_PARAMS']['OCV_FNAME']
            par_out = self.gen_param_interp_function( self.ocv_fn )
            if   par_out[1] == '1D' :
                self.ecm_params['ocv']['intp_func'], self.ecm_params['ocv']['dim'], d, d1 = par_out
                self.OCP = d
            elif par_out[1] == '2D' :
                self.ecm_params['ocv']['intp_func'], self.ecm_params['ocv']['dim'], d, D1, D2, d1, d2 = par_out
                self.OCP = d[:,0]
            self.xs  = d1

            # Ohmic Resistance data
            self.res_ohm_fn = fname_root + 'ecm/res/' + self.RunInput['ECM_PARAMS']['RES_OHMIC_FNAME']
            par_out = self.gen_param_interp_function( self.res_ohm_fn )
            if   par_out[1] == '1D' :
                self.ecm_params['res_ohm']['intp_func'], self.ecm_params['res_ohm']['dim'], d, d1 = par_out
            elif par_out[1] == '2D' :
                self.ecm_params['res_ohm']['intp_func'], self.ecm_params['res_ohm']['dim'], d, D1, D2, d1, d2 = par_out

            # Loop through for all RC elements
            self.res_fn = [ fname_root + 'ecm/res/' + self.RunInput['ECM_PARAMS']['RES_RC_FNAMES'][i_rc] for i_rc in range(self.ecmOrder) ]
            self.tau_fn = [ fname_root + 'ecm/tau/' + self.RunInput['ECM_PARAMS']['TAU_RC_FNAMES'][i_rc] for i_rc in range(self.ecmOrder) ]
            for i_rc in range(self.ecmOrder) :
                # Resistance data
                par_out = self.gen_param_interp_function( self.res_fn[i_rc] )
                self.ecm_params['res'][i_rc]['intp_func'] = par_out[0]
                self.ecm_params['res'][i_rc]['dim']       = par_out[1]

                # Time constant (tau) data
                par_out = self.gen_param_interp_function( self.tau_fn[i_rc] )
                self.ecm_params['tau'][i_rc]['intp_func'] = par_out[0]
                self.ecm_params['tau'][i_rc]['dim']       = par_out[1]

            # Cell capacity
            self.ecm_params['Ah_cap'] = RunInput['ECM_PARAMS']['ECM_OCV_CAP']

            # Initial state of charge
            self.SOC_0 = numpy.interp( self.V_init, self.OCP,self.xs )

        # Cell design properties
        self.cell_type = 'cyl' # or 'pouch'
        self.cell_mass = 0.0510530  # [kg]
        self.cell_dims = {'height':0.01}
        if (self.cell_type == 'cyl') : 
            self.cell_dims['height']    = 0.065 # [m]
            self.cell_dims['diameter']  = 0.018 # [m]
            self.cell_dims['volume']    = ( self.pi*(self.cell_dims['diameter']**2.)/4. ) * self.cell_dims['height'] # [m^3]
            self.cell_dims['surf_area'] = ( self.pi*(self.cell_dims['diameter']**2.)/4. )*2. + self.pi*(self.cell_dims['diameter']*self.cell_dims['height']) # [m^2]
        elif (self.cell_type == 'pouch') : 
            self.cell_dims['height'] = 0.080 # [m]
            self.cell_dims['width']  = 0.050 # [m]
            self.cell_dims['thick']  = 0.010 # [m]
            self.cell_dims['volume']    = self.cell_dims['height'] * self.cell_dims['width'] * self.cell_dims['thick'] # [m^3]
            self.cell_dims['surf_area'] = ( 2.*self.cell_dims['height'] * self.cell_dims['width'] + 2.*self.cell_dims['height'] * self.cell_dims['thick'] + 2.*self.cell_dims['thick'] * self.cell_dims['width']  ) # [m^2]
        else :
            print 'Wrong cell type'

        # Time step control variable initialization for CN-scheme
        self.delta_t = 1. # [s]

        ## --- Thermodynamic parameters --- ##
        # Themal
        self.C_p  = 880.0  # approx around 2013-Ji,...,C.Y. Wang Low-Temp Li-ion cell operation paper for 18650 cell
        self.h    = 10000. # high numbers (i.e.,10000.) provide near isothermal operation for model
        if self.modelPhysical :
            self.sa_frac = self.cell_dims['surf_area']/self.Area
            self.rho_avg = self.cell_mass / self.cell_dims['volume'] * ( self.L_n+self.L_s+self.L_p )
        else :
            self.sa_frac = 1.0
            self.rho_avg = self.cell_mass / self.cell_dims['volume']

        self.T_amb = T_init
#        self.T_dT  = RunInput['SIMULATION']['DELTA_T'][0]

        self.therm_on = 1

        ## KMM parameters for distributed models
        self.Ksm = RunInput['DIST_SOLVING']['K_DIST']
        self.sm_iter = RunInput['DIST_SOLVING']['SM_ITER_MAX']
        self.V_err_tol = RunInput['DIST_SOLVING']['DIST_V_TOL']
        self.I_kmm_dist = 1e-3
#        self.C_kmm_dist = 1e-3
        self.n_submod = RunInput['MODEL']['N_SUBMOD']

        ## --- Crank-Nicholson Control --- ##
        # Maximum sub iteration count
        self.cn_maxit = RunInput['TIMESTEPPING']['CN_MAX_ITERS']
        # Tolerance for iteration exit
        self.cn_tol   = RunInput['TIMESTEPPING']['CN_TOL']
        # Variable time stepping
        self.dOut_tol = RunInput['TIMESTEPPING']['DV_TOL'] # change in V for each time step |  .015 seems OK for CC discharges, .002 seems good for DCR
        self.dI_tol   = RunInput['MODEL']['RATE_NOM_CAP']/20
        self.min_start_dt = .01
        #self.min_end_dt   = 2.
        if RunInput['SIMULATION']['TEST_TYPE'] == 'Rate' :
            self.max_step = 1./60. # max normalized time step -> e.g. 1/100 is a max of 1# time step size over the full sim time -> a 3600sec test has a max step size of 36sec.
        else :
            self.max_step = 1./5.

        self.max_next      = 2. # i.e. dt_next <= dt_last*self.max_next, This helps to ensure dt size change stability
        self.max_next_cv   = 1.1
        self.max_rest_step = RunInput['TIMESTEPPING']['MAX_REST_STEP']
        self.max_subit     = 5

        self.volt_hold_tol  = 0.001
        self.max_volt_iters = 10
        ## kick outs
        f=.0001
        self.lo = 1.
        self.hi = 1.-f

    def gen_param_interp_function( self, fpath, x_scale=1.0, y_scale=1.0, z_scale=1.0 ) :
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
        
        Numbers should be the only values used in any element. This is constrained
        by us using numpy.loadtxt to load the data table file.
        """
        # Load the data table
        data_table = numpy.loadtxt( fpath, dtype='d', delimiter=',' )

        # Check for dimensions (handle either 1D or 2D interpolation)
        if len(data_table.shape) > 1 :   # Ensure more than one column used
            if data_table.shape[1] > 2 : # multiple Y values used (2D interpolation)
                d1 = data_table[1:,0] *x_scale
                d2 = data_table[0,1:] *y_scale
                d  = data_table[1:,1:]*z_scale

                # Check for increasing values (needed for interpolation class)
                if d1[1]<d1[0] :
                    d1 = numpy.flipud(d1)
                    d  = numpy.flipud(d )
                if d2[1]<d2[0] :
                    d2 = numpy.flipud(d2)
                    d  = numpy.fliplr(d )

                D1, D2 = numpy.meshgrid( d1, d2 )

                intp_func = scipy.interpolate.RectBivariateSpline( d1, d2, d )

                out = intp_func, '2D', d, D1, D2, d1, d2
            else : # 1D interpolation
                d1 = data_table[1:,0]*x_scale
                d  = data_table[1:,1]*y_scale

                # Check for increasing values
                if d1[1]<d1[0] :
                    d1 = numpy.flipud(d1)
                    d  = numpy.flipud(d )

                intp_func = scipy.interpolate.interp1d( d1, d )

                out = intp_func, '1D', d, d1
        else :
            print 'Data table format not correct!', '\n', 'Path:', fpath
            out = 'junk'

        return out

    def get_matl_properties( self, path ) :
        """
        Read in the material properties. Should be stored in data/matl_prop.txt,
        or something along those lines.
        """
        cfr = confreader.reader( path )
        self.matl_prop = cfr.conf_data

        self.proc_matl_prop()


    def get_des_properties( self, path ) :
        """
        Read in the cell design properties. Should be stored in data/des_prop.txt,
        or something along those lines.
        """
        cfr = confreader.reader( path )
        self.des_prop = cfr.conf_data

        self.proc_des_prop()


    def proc_matl_prop( self, ) :
        """
        After running the get_matl_prop...() function, this is run to extract and
        compute the required information.
        """
        mp = self.matl_prop

        self.sig_n = mp['SOLID_CONDUCTIVITY']['sig_n']
        self.sig_p = mp['SOLID_CONDUCTIVITY']['sig_p']

        self.sc_n = mp['SITE_CAPACITY']['sc_n']
        self.sc_p = mp['SITE_CAPACITY']['sc_p']

        self.Mm_n = mp['MOLEC_MASS']['molec_mass_n']
        self.Mm_p = mp['MOLEC_MASS']['molec_mass_p']

        self.rho_act_n = mp['DENS']['activmatl_n']
        self.rho_act_p = mp['DENS']['activmatl_p']

        self.rho_binders    = mp['DENS']['binders']
        self.rho_condaddtvs = mp['DENS']['cond_addtvs']

        self.num_act_n      = len(self.rho_act_n)
        self.num_act_p      = len(self.rho_act_p)
        self.num_binders    = len(self.rho_binders)
        self.num_condaddtvs = len(self.rho_condaddtvs)

        self.rho_electrolyte = mp['DENS']['electrolyte']

        self.t_plus = mp['ELECTROLYTE']['transference_number']

        self.c_s_n_max = (self.sc_n*3600/self.Faraday)*(self.rho_act_n[0]) # Max concentration in anode,   [mol/m^3]
        self.c_s_p_max = (self.sc_p*3600/self.Faraday)*(self.rho_act_p[0]) # Max concentration in cathode, [mol/m^3]


    def proc_des_prop( self, ) :
        """
        After running the get_matl_prop...(), proc_matl_prop...(), and
        get_des_prop...() functions, this is run to extract and compute the 
        required cell design related information.
        """
        dp = self.des_prop

        ## Particle radii
        self.R_s_n = dp['PARTICLE']['R_s_n']
        self.R_s_p = dp['PARTICLE']['R_s_p']

        ## Bruggeman correction terms for tortuosity
        self.brug_n = dp['TORTUOSITY']['brug_n']
        self.brug_s = dp['TORTUOSITY']['brug_s']
        self.brug_p = dp['TORTUOSITY']['brug_p']

        ## Geometry related items
        # Domain thicknesses
        self.L_n = dp['GEOMETRY']['L_n']
        self.L_s = dp['GEOMETRY']['L_s']
        self.L_p = dp['GEOMETRY']['L_p']

        # Coated area (for current density calculations)
        self.Area = dp['COATING']['HEIGHT']*dp['COATING']['LENGTH']
        ## Mass loading
        self.an_mL  = dp['MASS_LOADING']['an_mL'][0]
        self.cat_mL = dp['MASS_LOADING']['cat_mL'][0]
        self.AM_frac_n = dp['MASS_LOADING']['an_AMs'][0]
        self.AM_frac_p = dp['MASS_LOADING']['cat_AMs'][0]
        
        ## Volume fractions, TODO: need to change for multichem
        self.epsilon_e_n = dp['VOL_FRACS']['epsilon_e_n']
        self.epsilon_e_s = dp['VOL_FRACS']['epsilon_e_s']
        self.epsilon_e_p = dp['VOL_FRACS']['epsilon_e_p']

        self.epsilon_s_n = (( self.an_mL * self.AM_frac_n ) 
                              / ( self.rho_act_n[0] * self.L_n ))
        self.epsilon_s_p = (( self.cat_mL * self.AM_frac_p ) 
                              / ( self.rho_act_p[0] * self.L_p ))

        self.epsilon_f_n = sum( [(  self.an_mL *  dp['MASS_LOADING']['an_Bs'][i]) / (self.rho_binders[val-1]*  self.L_n) for i,val in enumerate(dp['MASS_LOADING']['an_B_nums']) ] )  \
                         + sum( [(  self.an_mL *  dp['MASS_LOADING']['an_CAs'][i]) / (self.rho_condaddtvs[val-1]*  self.L_n) for i,val in enumerate(dp['MASS_LOADING']['an_CA_nums']) ] )

        self.epsilon_f_p = sum( [(  self.cat_mL *  dp['MASS_LOADING']['cat_Bs'][i]) / (self.rho_binders[val-1]*  self.L_n) for i,val in enumerate(dp['MASS_LOADING']['cat_B_nums']) ] )  \
                         + sum( [(  self.cat_mL *  dp['MASS_LOADING']['cat_CAs'][i]) / (self.rho_condaddtvs[val-1]*  self.L_n) for i,val in enumerate(dp['MASS_LOADING']['cat_CA_nums']) ] )


    def gen_thermFactor( self, x, U, dUdx, bern_on ) :
        """
        Compute the activity correction term from the equilibrium potential data.
        One of two methods may be used. Dawn Bernardi, et al. have a technique 
        and essentially all other use a slightly different method. Both are 
        compute and an input control parameter in the conf file may be used to 
        switch between the two here.
        """
        if bern_on :
            tf = (0+ -(self.Faraday/self.R/298.15 * x * (1.-x)) * dUdx) / (1.-x)
            thermFactor = tf*(1./abs(numpy.mean(tf)))
        else :
            tf = (0+ -(self.Faraday/self.R/298.15 * x * (1.-x)) * dUdx)
            thermFactor = tf*(1./abs(numpy.mean(tf)))

        return thermFactor


    def get_fullstoichrange_ocv( self ) :
        """
        Compute the full ocv curve data beyond the stoich limits provide for alignment purposes.
        This is needed as sometimes the initial stoichs given may be say at 3.0V and 4.1V
        full cell voltages, however, one may actually be able to take the cell to
        2.5V and 4.2V based all of the equilibrium curve data provided for each
        half cell.
        """
        ## Determine new bottom cathode stoich
        # assume bottom of ocv is when anode is at 0.0 stoich
        anbottcap = self.sc_n*(self.Udat_n['botV_x']-self.Udat_n['min_x'])*(self.an_mL*self.AM_frac_n) *self.Area # [Ah], remaining Ah from bott cap anode value
        deltacat  = anbottcap / (self.sc_p*(self.cat_mL*self.AM_frac_p)*self.Area)

        # Update bottom stoichs to push anode to 0.0
        self.cat_botV_x = self.Udat_p['botV_x'] + deltacat
        self.an_botV_x  = self.Udat_n['min_x']

        min_cat_dod = (1. - self.Udat_p['botV_x'])*(self.sc_p*(self.cat_mL*self.AM_frac_p)*self.Area)

        ## Determine top stoichs such that U_full_top = 4.2V
        dod_an  = self.sc_n*(self.an_mL*self.AM_frac_n)*self.Area  * self.Udat_n['x']
        dod_cat_rough = self.sc_p*(self.cat_mL*self.AM_frac_p)*self.Area * self.Udat_p['x']
        dod_cat = dod_cat_rough.max() - dod_cat_rough - min_cat_dod

        max_dod = numpy.amin( [dod_cat.max(), dod_an.max()] )-0.001
        min_dod = numpy.amax( [dod_cat.min(), dod_an.min()] )+0.001

        dod = numpy.linspace( min_dod, max_dod, num=500, dtype='d' )

        Up  = numpy.interp( dod, numpy.flipud(dod_cat.flatten()), numpy.flipud(self.Udat_p['U'].flatten()) )
        Ua  = numpy.interp( dod, numpy.flipud(dod_an.flatten()) , numpy.flipud(self.Udat_n['U'].flatten())   )

        ocv = Up-Ua

        dod_42 = numpy.interp( 4.2, ocv, dod )

        Up_42  = numpy.interp( dod_42, numpy.flipud(dod_cat.flatten()), numpy.flipud(self.Udat_p['U'].flatten()) )
        Ua_42  = numpy.interp( dod_42, numpy.flipud(dod_an.flatten()) , numpy.flipud(self.Udat_n['U'].flatten())   )

        self.Ah_cap = dod_42
        self.Udat_p['topV_x'] = numpy.interp( Up_42, numpy.flipud(self.Udat_p['U'].flatten()), numpy.flipud(self.Udat_p['x'].flatten()) )
        self.Udat_n['topV_x']  = numpy.interp( Ua_42, self.Udat_n['U'].flatten() , self.Udat_n['x'].flatten() )


    def get_fullcell_ocv( self, V_init ) :
        """
        Setup the full cell ocv curve using the half cell potential data
        """
        nSoc = 200
        self.an_soc  = numpy.linspace(self.Udat_n['topV_x'], self.Udat_n['botV_x'], num=nSoc, dtype='d')
        self.cat_soc = numpy.linspace(self.Udat_p['topV_x'], self.Udat_p['botV_x'], num=nSoc, dtype='d')

        an_pot  = numpy.interp( self.an_soc , numpy.flipud(self.Udat_n['x']), numpy.flipud(self.Udat_n['U'])  )
        cat_pot = numpy.interp( self.cat_soc, self.Udat_p['x'], self.Udat_p['U'] )

        self.nom_cap = 2.15*(self.Area/0.070452)

        self.xs = numpy.linspace(1.,0.,nSoc, dtype='d')

        self.full_soc_an  = numpy.interp( self.Udat_n['x'], numpy.flipud(self.an_soc) , numpy.flipud(self.xs) )
        self.full_soc_cat = numpy.interp( self.Udat_p['x'], self.cat_soc, self.xs )

        self.Up = cat_pot
        self.Ua = an_pot

        self.OCP = self.Up-self.Ua

        if self.RunInput['MODEL']['ELECTRODE'] == 'full' or self.RunInput['MODEL']['ELECTRODE'] == 'anode' :
            soc_init = numpy.interp( V_init, numpy.flipud(self.OCP), numpy.flipud(self.xs) )
 
            Up_i = numpy.interp( soc_init, numpy.flipud(self.xs), numpy.flipud(self.Up) )
            Ua_i = numpy.interp( soc_init, numpy.flipud(self.xs), numpy.flipud(self.Ua) )

            self.V_init = Up_i - Ua_i

            self.theta_n0 = numpy.interp( Ua_i, an_pot , self.an_soc  )
            self.theta_p0 = numpy.interp( Up_i, numpy.flipud(cat_pot), numpy.flipud(self.cat_soc) )

            self.OCP_sim = numpy.concatenate( [ [V_init,], self.OCP[self.OCP<V_init] ] )
            self.xs_sim  = numpy.interp( self.OCP_sim, numpy.flipud(self.OCP), numpy.flipud(self.xs) ) 
            self.ah_sim  = (1.-self.xs_sim)*self.Ah_cap

        elif self.RunInput['MODEL']['ELECTRODE'] == 'cathode' :
            cat_x_init = numpy.interp( V_init, numpy.flipud(self.Udat_p['U']), numpy.flipud(self.Udat_p['x']) )
            self.theta_n0 = 0.5083

            Up_i = V_init
            Ua_i = numpy.interp( self.theta_n0, self.an_soc, self.Ua )

            self.theta_p0 = cat_x_init

            ocp_init = numpy.interp( Up_i, self.Up, self.OCP )

            self.OCP_sim = numpy.concatenate( [ [ocp_init,], self.OCP[self.OCP<ocp_init] ] )
            self.xs_sim  = numpy.interp( self.OCP_sim, self.OCP, self.xs ) 
            self.ah_sim  = (1.-self.xs_sim)*self.Ah_cap
        #else :
         #   print('Do not recognize the ELECTRODE parameter in the config file')
            # TODO: Need this to stop generate an error and stop simulation


#    def get_ecm_params( self, ) :
#        """
#        Load the Equivalent Circuit Model parameters. This includes the full cell
#        OCV curve, Resistances and Capacitances for each RC branch.
#        The OCV curve may be a table based on SOC and/or temperature.
#        The Resistances and Capacitances may be set as constants, or be tables
#        that are functions of SOC and/or Temperature.
#        """
#        # Load the OCV data
        

#    def get_Ds_params( self, ) :
#        """
#        Setup the solid phase diffusion coefficient data
#        """
#        if self.RunInput['SOLID_DIFFUSION']['VAR_DIFF_ANODE_ON'] :
#            if self.RunInput['SOLID_DIFFUSION']['AN_PROF_ON'] :
#                self.D_s_n = self.RunInput['SOLID_DIFFUSION']['Dsn']
#            else :
#                self.D_s_n = 1.0

#            self.Dsn_dat = numpy.loadtxt( fname_root+'/solid/diffusion/'+self.RunInput['SOLID_DIFFUSION']['DSN_FNAME'] )
#        else :
#            self.D_s_n = self.RunInput['SOLID_DIFFUSION']['Dsn'] # m^2/s


#        if self.RunInput['SOLID_DIFFUSION']['VAR_DIFF_CATHODE_ON'] :
#            if self.RunInput['SOLID_DIFFUSION']['CAT_PROF_ON'] :
#                self.D_s_p = self.RunInput['SOLID_DIFFUSION']['Dsp']
#            else :
#                self.D_s_p = 1.0

#            Dspdat  = numpy.loadtxt( fname_root+'/solid/diffusion/'+self.RunInput['SOLID_DIFFUSION']['DSP_FNAME'] )
#            dspOcv  = Dspdat[:,0]
#            dspDvals= Dspdat[:,1]

#            dspStoich = numpy.interp( dspOcv, self.cathode_refPot[:,1], self.cathode_refPot[:,0] )

#            newStoich  = numpy.linspace( dspStoich[0], dspStoich[-1], 500 )
#            newDspVals = numpy.interp( newStoich, dspStoich, dspDvals )
#            self.Dsp_dat  = numpy.reshape( numpy.array( [ newStoich, newDspVals ] ), [ len(newStoich), 2 ] )
#        else :
#            self.D_s_p = self.RunInput['SOLID_DIFFUSION']['Dsp'] # m^2/s

#        self.Dsp_Ea = 30000.0   # J/mol
#        self.Dsn_Ea = 30000.0   # J/mol
#        self.cathode_therm_on = 0
#        self.anode_therm_on   = 0

