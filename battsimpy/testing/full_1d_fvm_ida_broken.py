import numpy

import scipy.linalg
import scipy.optimize

from copy import deepcopy

from matplotlib import pyplot as plt
plt.style.use('classic')

from assimulo.solvers import IDA
from assimulo.problem import Implicit_Problem


from scipy.signal import wiener, filtfilt, butter, gaussian
from scipy.ndimage import filters

#import batteqns
#import params


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
#class FULL_1D( Implicit_Problem ) :
#    """
#    This is the Full 1D Model code
#    The solid phase diffusion is fully discretized at each point in x.
#        This enables studies where solid phase lithium ion diffusitiy is a 
#        function of the local solid phase lithium concentration both in the x 
#        direction as well as the radial particel direction i.e., D_s(c_s(x,r,t)) 
#    All parameters may be defined as lookup tables, or via function calls.
#        Electrolyte params: \kappa_e, D_e, f_{+/-} may all be defined as 
#                            functions of local electrolyte concentration 
#                            [c_e(x,t)] as well as lumped cell temperature.
#        Kinetic params: i_o may be a function of electrode surface concentration
#                        [c_s(x,r_ss,t)], and lumped cell temperature
#        
#    """
#    def __init__(self, confDat) :
#
#        # Setup the input parameters
#        self.confDat = confDat
#        self.Pdat    = { 'RunInput':self.confDat }
#        self.V_init  = 4.198 # [V]
#
#        self.schd = {}
#
#        # Run first Assimulo model setup, set the initial conditions
#        self.model_setup()
#        p = self.pars
#
#        self.t_end_now = 0.0
#
#        # Set the cell's coated area
#        self.Ac = p.Area
#
#        # Set temperatures
#        self.T_amb = 30.+273.15
#        self.T     = 30.+273.15  # Cell temperature, [K]
#
#        # Parameter state sensitivty control, 1 means the matrices will update 
#        # for each time step, as the states change
#        self.De_ce_on = 1
#        self.ke_ce_on = 1
#        self.Ds_cs_on = 1
#
#        ### System indices
#        self.ce_inds   = range( p.N )
#        self.ce_inds_r = numpy.reshape( self.ce_inds, [len(self.ce_inds),1] )
#        self.ce_inds_c = numpy.reshape( self.ce_inds, [1,len(self.ce_inds)] )
#
#        self.csa_inds = range( p.N, p.N + (p.Na*p.Nra) )
#        self.csa_inds_r = numpy.reshape( self.csa_inds, [len(self.csa_inds),1] )
#        self.csa_inds_c = numpy.reshape( self.csa_inds, [1,len(self.csa_inds)] )
#
#        self.csc_inds = range( p.N + (p.Na*p.Nra), p.N + (p.Na*p.Nra) + (p.Nc*p.Nrc) )
#        self.csc_inds_r = numpy.reshape( self.csc_inds, [len(self.csc_inds),1] )
#        self.csc_inds_c = numpy.reshape( self.csc_inds, [1,len(self.csc_inds)] )
#
#        self.T_ind = p.N + (p.Na*p.Nra) + (p.Nc*p.Nrc)
#
#        c_end = p.N + (p.Na*p.Nra) + (p.Nc*p.Nrc) + 1
#
#        self.ja_inds = range(c_end, c_end+p.Na)
#        self.ja_inds_r = numpy.reshape( self.ja_inds, [len(self.ja_inds),1] )
#        self.ja_inds_c = numpy.reshape( self.ja_inds, [1,len(self.ja_inds)] )
#
#        self.jc_inds = range(c_end+p.Na, c_end+p.Na +p.Nc)
#        self.jc_inds_r = numpy.reshape( self.jc_inds, [len(self.jc_inds),1] )
#        self.jc_inds_c = numpy.reshape( self.jc_inds, [1,len(self.jc_inds)] )
#        
#        self.pe_inds   = range( c_end+p.Na+p.Nc, c_end+p.Na+p.Nc +p.N )
#        self.pe_inds_r = numpy.reshape( self.pe_inds, [len(self.pe_inds),1] )
#        self.pe_inds_c = numpy.reshape( self.pe_inds, [1,len(self.pe_inds)] )
#
#        self.pe_a_inds = range( c_end+p.Na+p.Nc, c_end+p.Na+p.Nc +p.Na )
#        self.pe_a_inds_r = numpy.reshape( self.pe_a_inds, [len(self.pe_a_inds),1] )
#        self.pe_a_inds_c = numpy.reshape( self.pe_a_inds, [1,len(self.pe_a_inds)] )
#
#        self.pe_c_inds = range( c_end+p.Na+p.Nc +p.Na+p.Ns, c_end+p.Na+p.Nc +p.N )
#        self.pe_c_inds_r = numpy.reshape( self.pe_c_inds, [len(self.pe_c_inds),1] )
#        self.pe_c_inds_c = numpy.reshape( self.pe_c_inds, [1,len(self.pe_c_inds)] )
#
#        self.pa_inds = range( c_end+p.Na+p.Nc+p.N, c_end+p.Na+p.Nc+p.N +p.Na )
#        self.pa_inds_r = numpy.reshape( self.pa_inds, [len(self.pa_inds),1] )
#        self.pa_inds_c = numpy.reshape( self.pa_inds, [1,len(self.pa_inds)] )
#
#        self.pc_inds = range( c_end+p.Na+p.Nc+p.N+p.Na, c_end+p.Na+p.Nc+p.N+p.Na +p.Nc )
#        self.pc_inds_r = numpy.reshape( self.pc_inds, [len(self.pc_inds),1] )
#        self.pc_inds_c = numpy.reshape( self.pc_inds, [1,len(self.pc_inds)] )
#
#        # second set for manual jac version
#        c_end = 0
#        self.ja_inds2 = range(c_end, c_end+p.Na)
#        self.ja_inds_r2 = numpy.reshape( self.ja_inds2, [len(self.ja_inds2),1] )
#        self.ja_inds_c2 = numpy.reshape( self.ja_inds2, [1,len(self.ja_inds2)] )
#
#        self.jc_inds2 = range(c_end+p.Na, c_end+p.Na +p.Nc)
#        self.jc_inds_r2 = numpy.reshape( self.jc_inds2, [len(self.jc_inds2),1] )
#        self.jc_inds_c2 = numpy.reshape( self.jc_inds2, [1,len(self.jc_inds2)] )
#        
#        self.pe_inds2   = range( c_end+p.Na+p.Nc, c_end+p.Na+p.Nc +p.N )
#        self.pe_inds_r2 = numpy.reshape( self.pe_inds2, [len(self.pe_inds2),1] )
#        self.pe_inds_c2 = numpy.reshape( self.pe_inds2, [1,len(self.pe_inds2)] )
#
#        self.pe_a_inds2 = range( c_end+p.Na+p.Nc, c_end+p.Na+p.Nc +p.Na )
#        self.pe_a_inds_r2 = numpy.reshape( self.pe_a_inds2, [len(self.pe_a_inds2),1] )
#        self.pe_a_inds_c2 = numpy.reshape( self.pe_a_inds2, [1,len(self.pe_a_inds2)] )
#
#        self.pe_c_inds2 = range( c_end+p.Na+p.Nc +p.Na+p.Ns, c_end+p.Na+p.Nc +p.N )
#        self.pe_c_inds_r2 = numpy.reshape( self.pe_c_inds2, [len(self.pe_c_inds2),1] )
#        self.pe_c_inds_c2 = numpy.reshape( self.pe_c_inds2, [1,len(self.pe_c_inds2)] )
#
#        self.pa_inds2 = range( c_end+p.Na+p.Nc+p.N, c_end+p.Na+p.Nc+p.N +p.Na )
#        self.pa_inds_r2 = numpy.reshape( self.pa_inds2, [len(self.pa_inds2),1] )
#        self.pa_inds_c2 = numpy.reshape( self.pa_inds2, [1,len(self.pa_inds2)] )
#
#        self.pc_inds2 = range( c_end+p.Na+p.Nc+p.N+p.Na, c_end+p.Na+p.Nc+p.N+p.Na +p.Nc )
#        self.pc_inds_r2 = numpy.reshape( self.pc_inds2, [len(self.pc_inds2),1] )
#        self.pc_inds_c2 = numpy.reshape( self.pc_inds2, [1,len(self.pc_inds2)] )
#
#
#        self.var_inf = { 'x': {'c_e'    :{'size':p.N       ,'inds':self.ce_inds },
#                               'c_s_a'  :{'size':p.Na*p.Nra,'inds':self.csa_inds},
#                               'c_s_c'  :{'size':p.Nc*p.Nrc,'inds':self.csc_inds},
#                               'T'      :{'size':1         ,'inds':self.T_ind   }
#                             },
#                         'z': {'phi_a':{'size':p.Na,'inds':self.pa_inds},
#                               'phi_c':{'size':p.Nc,'inds':self.pc_inds},
#                               'phi_e':{'size':p.N ,'inds':self.pe_inds},
#                               'ja'   :{'size':p.Na,'inds':self.ja_inds},
#                               'jc'   :{'size':p.Nc,'inds':self.jc_inds}
#                             }
#                       }
#
#        # Build the initial system matrices
#        self.cs_mats()
##
##        self.build_Ace_mat(self.y00)
##        self.build_Ape_mat(self.y00)
##        self.build_Bpe_mat(self.y00)
#
##        self.build_Cio(self.y00, self.y00[self.csa_inds[:p.Na]], self.y00[self.csc_inds[:p.Nc]])
#
#        self.phie_mats()
#        self.phis_mats()
#
#        ### Matrices for thermal calcs (gradient operators)
#        self.Ga, self.Gc, self.G = batteqns.grad_mat( p.Na, p.x_m_a ), batteqns.grad_mat( p.Nc, p.x_m_c ), batteqns.grad_mat( p.N, p.x_m )
#
#        # Initialize the C arrays for the heat generation (these are useful for the Jacobian)
##        junkQ = self.calc_heat( y0, numpy.zeros(p.Na), numpy.zeros(p.Nc), p.uref_a( y0[self.csa_inds[:p.Na]]/p.csa_max ), p.uref_c( y0[self.csc_inds[:p.Nc]]/p.csc_max ) )
##
##        # Kinetic C array (also useful for the Jacobian)
##        csa_ss = y0[self.csa_inds[:p.Na]]
##        csc_ss = y0[self.csc_inds[:p.Nc]]
##        ce = y0[self.ce_inds]
##        T  = y0[self.T_ind]
##        self.C_ioa = 2.0*p.ioa_interp(csa_ss/p.csa_max, T, grid=False).flatten()/p.F * numpy.sqrt( ce[:p.Na ]/p.ce_nom * (1.0 - csa_ss/p.csa_max) * (csa_ss/p.csa_max) )
##        self.C_ioc = 2.0*p.ioc_interp(csc_ss/p.csc_max, T, grid=False).flatten()/p.F * numpy.sqrt( ce[-p.Nc:]/p.ce_nom * (1.0 - csc_ss/p.csc_max) * (csc_ss/p.csc_max) )
#
#
#    def buildpars( self ) :
#        """
#        self.pars will be built here
#        """
#        self.pars = params.params()
#        self.pars.buildpars( self.V_init, self.Pdat )
#
#
#    def model_setup(self) :
#
#        self.buildpars()
#
#        y0, yd0 = self.set_initial_states(1,'')
#
#        self.y00, self.yd00 = y0, yd0
#
#        Implicit_Problem.__init__(self,y0=y0,yd0=yd0,name='FULL_1D_IDA')
#
#
#    def get_input( self, inp_typ, inp_val ) :
#        """
#        Setup the input variable for the model during the simulation based on 
#        the test schedule.
#        """
#        if inp_typ == 'Rest' :
#            self.inp = 0.0
#            self.pars.inp_bc = 'curr'
#            self.pars.rest = 1
#
#        elif inp_typ == 'Crate' : 
#            self.inp = (-inp_val*self.confDat['MODEL']['RATE_NOM_CAP']) / self.pars.Area
#            self.pars.inp_bc = 'curr'
#            self.pars.rest = 0
#
#        elif inp_typ == 'Current' :
#            self.inp = (-inp_val) / self.pars.Area
#            self.pars.inp_bc = 'curr'
#            self.pars.rest = 0
#
#         TODO - CV mode
#        elif inp_typ == 'Volt' : 
#            inp = inp_val
#            self.pars.inp_bc = 'volt'
#            self.pars.rest = 0
#
#         TODO - CP mode
#        elif inp_typ == 'Power' : 
#            inp = inp_val
#            self.pars.inp_bc = 'power'
#            self.pars.rest = 0
#
##        return inp

#
#    def set_iapp( self, I_app ) :
#        self.i_app = I_app / self.Ac
#
#
#    def phie_mats(self,) :
#        """
#        Electrolyte constant B_ce matrix
#        """
#        p = self.pars
#
#        Ba = [ (1.-p.t_plus)*asa/ea for ea, asa in zip(p.eps_a_vec,p.as_a) ]
#        Bs = [  0.0                for i in range(p.Ns) ]
#        Bc = [ (1.-p.t_plus)*asc/ec for ec, asc in zip(p.eps_c_vec,p.as_c) ]
#
#        self.B_ce = numpy.diag( numpy.array(Ba+Bs+Bc, dtype='d') )
#
#        Bap = [ asa*p.F for asa in p.as_a  ]
#        Bsp = [   0.0 for i   in range(p.Ns) ]
#        Bcp = [ asc*p.F for asc in p.as_c  ]
#
#        self.B2_pe = numpy.diag( numpy.array(Bap+Bsp+Bcp, dtype='d') )
#
#
#    def phis_mats(self,) :
#        """
#        Solid phase parameters and j vector matrices
#        """
#        p = self.pars
#
#        self.A_ps_a = batteqns.flux_mat_builder( p.Na, p.x_m_a, numpy.ones_like(p.vols_a), p.sig_a_eff )
#        self.A_ps_c = batteqns.flux_mat_builder( p.Nc, p.x_m_c, numpy.ones_like(p.vols_c), p.sig_c_eff )
#
#        # Grounding form for BCs (was only needed during testing, before BVK was incorporated for coupling
##        self.A_ps_a[-1,-1] = 2*self.A_ps_a[-1,-1]
##        self.A_ps_c[ 0, 0] = 2*self.A_ps_c[ 0, 0]
#
#        Baps = numpy.array( [ asa*p.F*dxa for asa,dxa in zip(p.as_a, p.vols_a) ], dtype='d' )
#        Bcps = numpy.array( [ asc*p.F*dxc for asc,dxc in zip(p.as_c, p.vols_c) ], dtype='d' )
#
#        self.B_ps_a = numpy.diag( Baps )
#        self.B_ps_c = numpy.diag( Bcps )
#
#        self.B2_ps_a = numpy.zeros( p.Na, dtype='d' )
#        self.B2_ps_a[ 0] = -1.
#        self.B2_ps_c = numpy.zeros( p.Nc, dtype='d' )
#        self.B2_ps_c[-1] = -1.
#
#
#    def cs_mats(self,) :
#        """
#        Solid phase diffusion model
#        """
#        p = self.pars
#
#        ## 1D spherical diffusion model
#        # A_cs pre build
#        self.A_csa_single = batteqns.flux_mat_builder( p.Nra, p.r_m_a, p.vols_ra_m, p.Dsa*(p.r_e_a**2) )
#        self.A_csc_single = batteqns.flux_mat_builder( p.Nrc, p.r_m_c, p.vols_rc_m, p.Dsc*(p.r_e_c**2) )
#
#        # A_cs build up to the stacked full cs size (Nr and Nx)
#        b = [self.A_csa_single]*p.Na
#        self.A_cs_a = scipy.linalg.block_diag( *b )
#        b = [self.A_csc_single]*p.Nc
#        self.A_cs_c = scipy.linalg.block_diag( *b )
#
#        # B_cs and C_cs are constant (i.e., are not state-dependent)
#        self.B_csa_single = numpy.array( [ 0. for i in range(p.Nra-1) ]+[-1.*p.r_e_a[-1]**2/p.vols_ra_m[-1]], dtype='d' )
#        self.B_csc_single = numpy.array( [ 0. for i in range(p.Nrc-1) ]+[-1.*p.r_e_c[-1]**2/p.vols_rc_m[-1]], dtype='d' )
#
#        b = [self.B_csa_single]*p.Na
#        self.B_cs_a = scipy.linalg.block_diag( *b ).T
#        b = [self.B_csc_single]*p.Nc
#        self.B_cs_c = scipy.linalg.block_diag( *b ).T
#
#        # Particle surface concentration
#        h_na  = p.r_e_a[-1] - p.r_m_a[-1]
#        h_n1a = p.r_m_a[-1] - p.r_m_a[-2]
#
#        h_nc  = p.r_e_c[-1] - p.r_m_c[-1]
#        h_n1c = p.r_m_c[-1] - p.r_m_c[-2]
#
#        self.a_n_a, self.b_n_a, self.c_n_a = batteqns.right_side_coeffs( h_na, h_n1a )
#        self.a_n_c, self.b_n_c, self.c_n_c = batteqns.right_side_coeffs( h_nc, h_n1c )
#
#        self.C_cs_a_single = numpy.array( [0. for i in range(p.Nra-2)]+[-self.a_n_a/self.c_n_a, -self.b_n_a/self.c_n_a], dtype='d' )
#        self.C_cs_c_single = numpy.array( [0. for i in range(p.Nrc-2)]+[-self.a_n_c/self.c_n_c, -self.b_n_c/self.c_n_c], dtype='d' )
#
#        self.C_cs_a = scipy.linalg.block_diag( *[self.C_cs_a_single]*p.Na )
#        self.C_cs_c = scipy.linalg.block_diag( *[self.C_cs_c_single]*p.Nc )
#
#        self.C_cs_a_avg = scipy.linalg.block_diag( *[1./((1./3.)*p.Rp_a**3)*p.vols_ra_m]*p.Na )
#        self.C_cs_c_avg = scipy.linalg.block_diag( *[1./((1./3.)*p.Rp_c**3)*p.vols_rc_m]*p.Nc )
#
#        self.C_cs_a_mean = 1./p.La*p.vols_a.dot(self.C_cs_a_avg)
#        self.C_cs_c_mean = 1./p.Lc*p.vols_c.dot(self.C_cs_c_avg)
#
#        # Particle core concentration
#        h_na  = p.r_e_a[0] - p.r_m_a[0]
#        h_n1a = p.r_m_a[1] - p.r_m_a[0]
#
#        h_nc  = p.r_e_c[0] - p.r_m_c[0]
#        h_n1c = p.r_m_c[1] - p.r_m_c[0]
#
#        a_n_a, b_n_a, c_n_a = batteqns.left_side_coeffs( h_na, h_n1a )
#        a_n_c, b_n_c, c_n_c = batteqns.left_side_coeffs( h_nc, h_n1c )
#
#        C_cso_a_single = numpy.array( [-b_n_a/a_n_a, -c_n_a/a_n_a] + [0. for i in range(p.Nra-2)], dtype='d' )
#        C_cso_c_single = numpy.array( [-b_n_c/a_n_c, -c_n_c/a_n_c] + [0. for i in range(p.Nrc-2)], dtype='d' )
#
#        self.C_cso_a = scipy.linalg.block_diag( *[C_cso_a_single]*p.Na )
#        self.C_cso_c = scipy.linalg.block_diag( *[C_cso_c_single]*p.Nc )
#
#        # D_cs prelim values, note this is Ds(cs) dependent and therefore requires updating for state dependent Ds
#        self.D_cs_a = -1.0/(p.Dsa*self.c_n_a)*numpy.eye( p.Na )
#        self.D_cs_c = -1.0/(p.Dsc*self.c_n_c)*numpy.eye( p.Nc )
#
#    # cs mats
#    def update_cs_mats( self, csa, csc, csa_ss, csc_ss, csa_o, csc_o ) :
#
#        p = self.pars
#
#        Acsa_list = [ [] for i in range(p.Na) ]
#        Acsc_list = [ [] for i in range(p.Nc) ]
#
#        Dsa_ss = [ 0. for i in range(p.Na) ]
#        Dsc_ss = [ 0. for i in range(p.Nc) ]
#
#        for ia in range(p.Na) :
#
#            csa_m = csa[ia*p.Nra:(ia+1)*p.Nra]
#            csa_e = numpy.array( [csa_o[ia]] + [ 0.5*(csa_m[i+1]+csa_m[i]) for i in range(p.Nra-1) ] + [csa_ss[ia]] )
#            Ua_e  = p.uref_a( csa_e/p.csa_max )
#            Dsa_e = p.Dsa_intp( Ua_e )
#
#            Acsa_list[ia] = batteqns.flux_mat_builder( p.Nra, p.r_m_a, p.vols_ra_m, Dsa_e*(p.r_e_a**2) )
#
#            Dsa_ss[ia] = Dsa_e[-1]
#
#        for ic in range(p.Nc) :
#
#            csc_m = csc[ic*p.Nrc:(ic+1)*p.Nrc]
#            csc_e = numpy.array( [csc_o[ic]] + [ 0.5*(csc_m[i+1]+csc_m[i]) for i in range(p.Nrc-1) ] + [csc_ss[ic]] )
#            Uc_e  = p.uref_c( csc_e/p.csc_max )
#            Dsc_e = p.Dsc_intp( Uc_e )
#
#            Acsc_list[ic] = batteqns.flux_mat_builder( p.Nrc, p.r_m_c, p.vols_rc_m, Dsc_e*(p.r_e_c**2) )
#
#            Dsc_ss[ic] = Dsc_e[-1]
#
#    #        b = self.A_csa_single.reshape(1,Nra,Nra).repeat(Na,axis=0)
#        self.A_cs_a = scipy.linalg.block_diag( *Acsa_list )
#        self.A_cs_c = scipy.linalg.block_diag( *Acsc_list )
#
#        self.D_cs_a = numpy.diag( -1.0/(numpy.array(Dsa_ss)*self.c_n_a) )
#        self.D_cs_c = numpy.diag( -1.0/(numpy.array(Dsc_ss)*self.c_n_c) )
#
#
#    ## Define c_e functions
#    def build_Ace_mat( self, c, T ) :
#
#        p = self.pars
#
#        D_eff = self.Diff_ce( c, T )
#
#        A = p.K_m.dot( batteqns.flux_mat_builder( p.N, p.x_m, p.vols, D_eff ) )
#
#        return A
#
#    def Diff_ce( self, c, T, mid_on=0, eps_off=0 ) :
#
#        p = self.pars
#
#        D_ce = p.De_intp( c, T, grid=False ).flatten()
#        
#        if eps_off :
#            D_mid = D_ce
#        else :
#            D_mid = D_ce * p.eps_eff
#
#        if mid_on :
#            D_out = D_mid
#        else :
#            if type(c) == float :
#                D_out = D_mid
#            else :
#                D_out = batteqns.mid_to_edge( D_mid, p.x_e )
#
#        return D_out
#
#    ## Define phi_e functions
#    def build_Ape_mat( self, c, T ) :
#        
#        p = self.pars
#
#        k_eff = self.kapp_ce( c, T )
#
#        A = batteqns.flux_mat_builder( p.N, p.x_m, p.vols, k_eff )
#
#        A[-1,-1] = 2*A[-1,-1] # BC update for phi_e = 0
#
#        return A
#
#    def build_Bpe_mat( self, c, T ) :
#
#        p = self.pars
#
#        gam = 2.*(1.-p.t_plus)*p.R_gas*T / p.F
#
#        k_eff = self.kapp_ce( c, T )
#
#        c_edge = batteqns.mid_to_edge( c, p.x_e )
#
#        B1 = batteqns.flux_mat_builder( p.N, p.x_m, p.vols, k_eff*gam/c_edge )
#
#        return B1
#
#    def kapp_ce( self, c, T, mid_on=0, eps_off=0 ) :
#
#        p = self.pars
#
#        k_ce = 1e-1*p.ke_intp( c, T, grid=False ).flatten() # 1e-1 converts from mS/cm to S/m (model uses SI units)
#
#        if eps_off :
#            k_mid = k_ce
#        else :
#            k_mid = k_ce * p.eps_eff
#
#        if mid_on :
#            k_out = k_mid
#        else :
#            if type(c) == float :
#                k_out = k_mid
#            else :
#                k_out = batteqns.mid_to_edge( k_mid, p.x_e )
#
#        return k_out
#
#    def build_Bjac_mat( self, eta, a, b ) :
#            
#        d = a*numpy.cosh( b*eta )*b
#
#        return numpy.diag( d )
#
#    def build_BjT_mat( self, T, a, b ) :
#            
#        d = a*numpy.cosh( b/T )*(-b/T**2)
#
#        return d
#        
#    def get_voltage( self, y ) :
#        """
#        Return the cell potential
#        """
#        pc = y[self.pc_inds]
#        pa = y[self.pa_inds]
#
#        Vcell = pc[-1] - pa[0]
#
#        return Vcell
#        
#    def get_eta_uref( self, csa, csc, ja_rxn, jc_rxn, phi_s_a, phi_s_c, phi ) :
#
#        p = self.pars
#
#        csa_ss = (self.C_cs_a.dot(csa)).flatten() + (self.D_cs_a.dot(ja_rxn)).flatten()
#        csc_ss = (self.C_cs_c.dot(csc)).flatten() + (self.D_cs_c.dot(jc_rxn)).flatten()
#
#        Uref_a = p.uref_a( csa_ss/p.csa_max ) # anode   equilibrium potential
#        Uref_c = p.uref_c( csc_ss/p.csc_max ) # cathode equilibrium potential
#
#        eta_a  = phi_s_a - phi[:p.Na]  - Uref_a  # anode   overpotential
#        eta_c  = phi_s_c - phi[-p.Nc:] - Uref_c  # cathode overpotential
#
#        return eta_a, eta_c, Uref_a, Uref_c, csa_ss, csc_ss
#
#    def update_Cio( self, csa_ss, csc_ss, ce, T ) :
#
#        p = self.pars
#
#        self.C_ioa = 2.0*p.ioa_interp(csa_ss/p.csa_max, T, grid=False).flatten()/p.F * numpy.sqrt( ce[:p.Na ]/p.ce_nom * (1.0 - csa_ss/p.csa_max) * (csa_ss/p.csa_max) )
#        self.C_ioc = 2.0*p.ioc_interp(csc_ss/p.csc_max, T, grid=False).flatten()/p.F * numpy.sqrt( ce[-p.Nc:]/p.ce_nom * (1.0 - csc_ss/p.csc_max) * (csc_ss/p.csc_max) )
#
#    def calc_heat( self, y, eta_a, eta_c, Uref_a, Uref_c ) :
#        """
#        Return the total integrated heat source across the cell sandwich
#        """
#        p = self.pars
#
#        ce      = y[ self.ce_inds  ]
#        csa     = y[ self.csa_inds ]
#        csc     = y[ self.csc_inds ]
#        ja      = y[ self.ja_inds  ]
#        jc      = y[ self.jc_inds  ]
#        phi     = y[ self.pe_inds  ]
#        phi_s_a = y[ self.pa_inds  ]
#        phi_s_c = y[ self.pc_inds  ]
#        T = y[self.T_ind]
#
#        # Gradients for heat calc
#        dphi_s_a = numpy.gradient( phi_s_a ) / numpy.gradient( p.x_m_a )
#        dphi_s_c = numpy.gradient( phi_s_c ) / numpy.gradient( p.x_m_c )
#
#        dphi = numpy.gradient( phi ) / numpy.gradient( p.x_m )
#
#        dlnce = 1./ce * ( numpy.gradient(ce) / numpy.gradient( p.x_m ) )
#
#        kapp_eff_m = self.kapp_ce( ce, T, mid_on=1 ) # kapp_eff at the node points (middle of control volume, rather than edge)
#
#        # Reaction kinetics heat
#        C_ra = (p.vols_a*p.F*p.as_a)
#        C_rc = (p.vols_c*p.F*p.as_c)
#
#        Q_rxn_a = C_ra.dot( ja*eta_a )
#        Q_rxn_c = C_rc.dot( jc*eta_c )
#        Q_rxn = Q_rxn_a + Q_rxn_c
#
#        csa_mean = self.C_cs_a_avg.dot(csa)
#        csc_mean = self.C_cs_c_avg.dot(csc)
#        Uam = p.uref_a( csa_mean/p.csa_max )
#        Ucm = p.uref_c( csc_mean/p.csc_max )
#
#        eta_conc_a = Uref_a-Uam
#        eta_conc_c = Uref_c-Ucm
#
#        Q_conc_a = C_ra.dot( eta_conc_a*ja )
#        Q_conc_c = C_rc.dot( eta_conc_c*jc )
#
#        Q_conc = Q_conc_a + Q_conc_c
#
#        # Ohmic heat in electrolyte and solid
#        C_pe = (p.vols.dot( numpy.diag(kapp_eff_m*dphi).dot(self.G) ) + 
#                p.vols.dot( numpy.diag(2*kapp_eff_m*p.R_gas*T/p.F*(1.-p.t_plus)*dlnce).dot(self.G) ))
#
#        Q_ohm_e = C_pe.dot(phi)
#
#        C_pa = p.vols_a.dot( numpy.diag(p.sig_a_eff*dphi_s_a).dot(self.Ga) )
#        C_pc = p.vols_c.dot( numpy.diag(p.sig_c_eff*dphi_s_c).dot(self.Gc) )
#
#        Q_ohm_s = C_pa.dot(phi_s_a) + C_pc.dot(phi_s_c)
#
#        Q_ohm = Q_ohm_e + Q_ohm_s
#
#        # Entropic heat
#        ## ??
#
#        # Total heat
#        Q_tot = Q_ohm + Q_rxn + Q_conc
#
#        self.C_q_pe = C_pe
#        self.C_q_pa = C_pa
#        self.C_q_pc = C_pc
#        self.C_q_na = C_ra*ja
#        self.C_q_nc = C_rc*jc
#        self.C_q_ja = C_ra*eta_a + C_ra*eta_conc_a
#        self.C_q_jc = C_rc*eta_c + C_rc*eta_conc_c
#
#        return Q_tot
#
#    ## Define system equations
#    def res( self, t, y, yd ) :
#
#        p = self.pars
#
#        ## Parse out the states
#        # E-lyte conc
#        ce     = y[ self.ce_inds]
#        c_dots = yd[self.ce_inds]
#
#        # Solid conc a:anode, c:cathode
#        csa    = y[ self.csa_inds]
#        csc    = y[ self.csc_inds]
#        csa_dt = yd[self.csa_inds]
#        csc_dt = yd[self.csc_inds]
#
#        # Reaction (Butler-Volmer Kinetics)
#        ja_rxn = y[self.ja_inds]
#        jc_rxn = y[self.jc_inds]
#
#        # E-lyte potential
#        phi = y[self.pe_inds]
#
#        # Solid potential
#        phi_s_a = y[self.pa_inds]
#        phi_s_c = y[self.pc_inds]
#
#        # Thermal
#        T    = y[ self.T_ind]
#        T_dt = yd[self.T_ind]
#
#        ## Grab state dependent matrices
#        # For E-lyte conc and potential (i.e., De(ce), kapp_e(ce))
#        A_ce = self.build_Ace_mat( ce, T )
#        A_pe = self.build_Ape_mat( ce, T )
#        B_pe = self.build_Bpe_mat( ce, T )
#
#        ## Compute extra variables
#        # For the reaction kinetics
#        eta_a, eta_c, Uref_a, Uref_c, csa_ss, csc_ss = self.get_eta_uref( csa, csc, ja_rxn, jc_rxn, phi_s_a, phi_s_c, phi )
#
#        # For Solid conc Ds
#        csa_o = (self.C_cso_a.dot(csa)).flatten()
#        csc_o = (self.C_cso_c.dot(csc)).flatten()
#
#        self.update_cs_mats( csa, csc, csa_ss, csc_ss, csa_o, csc_o )
#
#        # For kinetics, the io param is now conc dependent
#        self.update_Cio( csa_ss, csc_ss, ce, T )
#
#        Q_in = self.calc_heat( y, eta_a, eta_c, Uref_a, Uref_c )
#
#        Q_out = p.h*p.Aconv*(T - self.T_amb)
#
#        ja = self.C_ioa * numpy.sinh( 0.5*p.F/(p.R_gas*T)*eta_a )
#        jc = self.C_ioc * numpy.sinh( 0.5*p.F/(p.R_gas*T)*eta_c )
#
#        j = numpy.concatenate( [ ja_rxn, numpy.zeros(p.Ns), jc_rxn ] )
#
#        ## Compute the residuals
#        # Time deriv components
#        r1 = c_dots - ( ((A_ce.dot(ce)).flatten() + (self.B_ce.dot(j)).flatten()) ) # E-lyte conc
#
#        r2 = csa_dt - (self.A_cs_a.dot(csa).flatten() + self.B_cs_a.dot(ja_rxn).flatten()) # Anode   conc
#        r3 = csc_dt - (self.A_cs_c.dot(csc).flatten() + self.B_cs_c.dot(jc_rxn).flatten()) # Cathode conc
#
#        r4 = T_dt - 1./(p.rho*p.Cp)*(Q_in - Q_out)
#
#        # Algebraic components
#        r5 = ja_rxn - ja
#        r6 = jc_rxn - jc 
#
#        r7 = A_pe.dot(phi).flatten() - B_pe.dot(ce).flatten() + self.B2_pe.dot(j).flatten() # E-lyte potential
#
#        r8 = self.A_ps_a.dot(phi_s_a).flatten() - self.B_ps_a.dot(ja_rxn).flatten() - self.B2_ps_a*self.i_app # Anode   potential
#        r9 = self.A_ps_c.dot(phi_s_c).flatten() - self.B_ps_c.dot(jc_rxn).flatten() + self.B2_ps_c*self.i_app # Cathode potential
#
#        res_out = numpy.concatenate( [r1, r2, r3, [r4], r5, r6, r7, r8, r9] )
#
#        return res_out
#
#    def jac( self, c, t, y, yd ) :
#
#        p = self.pars
#
#        ### Setup 
#        ## Parse out the states
#        # E-lyte conc
#        ce     = y[ self.ce_inds]
#
#        # Solid conc a:anode, c:cathode
#        csa    = y[ self.csa_inds]
#        csc    = y[ self.csc_inds]
#
#        # Reaction (Butler-Volmer Kinetics)
#        ja_rxn = y[self.ja_inds]
#        jc_rxn = y[self.jc_inds]
#
#        # E-lyte potential
#        phi = y[self.pe_inds]
#
#        # Solid potential
#        phi_s_a = y[self.pa_inds]
#        phi_s_c = y[self.pc_inds]
#
#        # Temp
#        T = y[self.T_ind]
#
#        ## Grab state dependent matrices
#        # For E-lyte conc and potential (i.e., De(ce), kapp_e(ce))
#        A_ce = self.build_Ace_mat( ce, T )
#        A_pe = self.build_Ape_mat( ce, T )
#        B_pe = self.build_Bpe_mat( ce, T )
#
#        ## Compute extra variables
#        # For the reaction kinetics
#        eta_a, eta_c, Uref_a, Uref_c, csa_ss, csc_ss = self.get_eta_uref( csa, csc, ja_rxn, jc_rxn, phi_s_a, phi_s_c, phi )
#
#        ### Build the Jac matrix
#        ## Self coupling
#        A_dots = numpy.diag( [1*c for i in range(p.num_diff_vars)] )
#        j_c    = A_dots - scipy.linalg.block_diag( A_ce, self.A_cs_a, self.A_cs_c, [-p.h*p.Aconv/p.rho/p.Cp] )
#
#        Bjac_a = self.build_Bjac_mat( eta_a, self.C_ioa, 0.5*p.F/(p.R_gas*T) )
#        Bjac_c = self.build_Bjac_mat( eta_c, self.C_ioc, 0.5*p.F/(p.R_gas*T) )
#
#        BjT_a = self.build_BjT_mat( T, self.C_ioa, 0.5*p.F/(p.R_gas)*eta_a )
#        BjT_c = self.build_BjT_mat( T, self.C_ioc, 0.5*p.F/(p.R_gas)*eta_c )
#
##        dcss_dcs_a = self.C_cs_a_single
##        dcss_dcs_c = self.C_cs_c_single
#
#        dcss_dja = numpy.diagonal( self.D_cs_a )
#        dcss_djc = numpy.diagonal( self.D_cs_c )
#
#        dU_csa_ss = (1.0/p.csa_max)*p.duref_a(csa_ss/p.csa_max)
#        dU_csc_ss = (1.0/p.csc_max)*p.duref_c(csc_ss/p.csc_max)
#
#        DUDcsa_ss = numpy.diag( dU_csa_ss )
#        DUDcsc_ss = numpy.diag( dU_csc_ss )
#
#        A_ja = numpy.diag(numpy.ones(p.Na)) - (Bjac_a.dot(-1.0*DUDcsa_ss*1.0)).dot( self.D_cs_a )
#        A_jc = numpy.diag(numpy.ones(p.Nc)) - (Bjac_c.dot(-1.0*DUDcsc_ss*1.0)).dot( self.D_cs_c )
#
#        j = scipy.linalg.block_diag( j_c, A_ja, A_jc, A_pe, self.A_ps_a, self.A_ps_c )
#
#        ## Cross coupling
#        # c_e: j coupling back in
#        j[ numpy.ix_(self.ce_inds, self.ja_inds) ] = -self.B_ce[:, :p.Na ]
#        j[ numpy.ix_(self.ce_inds, self.jc_inds) ] = -self.B_ce[:, -p.Nc:]
#
#        # cs_a: j coupling
#        j[ numpy.ix_(self.csa_inds, self.ja_inds) ] = -self.B_cs_a
#        # cs_c: j coupling
#        j[ numpy.ix_(self.csc_inds, self.jc_inds) ] = -self.B_cs_c
#
#        a_coeff = 2.0*self.C_q_na*(-1.0)*dU_csa_ss
#        Ca_T = numpy.array([ self.C_cs_a_single*ac for ac in a_coeff ]).flatten()
#        c_coeff = 2.0*self.C_q_nc*(-1.0)*dU_csc_ss
#        Cc_T = numpy.array([ self.C_cs_c_single*cc for cc in c_coeff ]).flatten()
#
#        # T
#        j[self.T_ind,self.ja_inds]  = -1./(p.rho*p.Cp)*(self.C_q_ja + 2.0*(self.C_q_na*(-1.0)*dU_csa_ss*dcss_dja))
#        j[self.T_ind,self.jc_inds]  = -1./(p.rho*p.Cp)*(self.C_q_jc + 2.0*(self.C_q_nc*(-1.0)*dU_csc_ss*dcss_djc))
#        j[self.T_ind,self.pe_inds]  = -1./(p.rho*p.Cp)*(self.C_q_pe + numpy.array( list(self.C_q_na)+[0. for i in range(p.Ns)]+list(self.C_q_nc) )*(-1.0))
#        j[self.T_ind,self.pa_inds]  = -1./(p.rho*p.Cp)*(self.C_q_pa + self.C_q_na*(1.0))
#        j[self.T_ind,self.pc_inds]  = -1./(p.rho*p.Cp)*(self.C_q_pc + self.C_q_nc*(1.0))
#        j[self.T_ind,self.csa_inds] = -1./(p.rho*p.Cp)*(Ca_T)
#        j[self.T_ind,self.csc_inds] = -1./(p.rho*p.Cp)*(Cc_T)
#
#        j[self.ja_inds,self.T_ind] = -BjT_a
#        j[self.jc_inds,self.T_ind] = -BjT_c
#
#
#        # j_a: pe, pa, csa  coupling
#        j[numpy.ix_(self.ja_inds, self.pa_inds  )] = -Bjac_a*( 1.0)
#        j[numpy.ix_(self.ja_inds, self.pe_a_inds)] = -Bjac_a*(-1.0)
#        j[numpy.ix_(self.ja_inds, self.csa_inds )] = -(Bjac_a.dot(-1.0*DUDcsa_ss*1.0)).dot( self.C_cs_a )
#
#        # j_c: pe, pc, csc  coupling         
#        j[numpy.ix_(self.jc_inds, self.pc_inds  )] = -Bjac_c*( 1.0)
#        j[numpy.ix_(self.jc_inds, self.pe_c_inds)] = -Bjac_c*(-1.0)
#        j[numpy.ix_(self.jc_inds, self.csc_inds )] = -(Bjac_c.dot(-1.0*DUDcsc_ss*1.0)).dot( self.C_cs_c )
#
#        # phi_e: ce coupling into phi_e equation
#        j[numpy.ix_(self.pe_inds,self.ce_inds)] = -B_pe
#        j[numpy.ix_(self.pe_inds,self.ja_inds)] = self.B2_pe[:,:p.Na]
#        j[numpy.ix_(self.pe_inds,self.jc_inds)] = self.B2_pe[:,-p.Nc:]
#
#        # phi_s_a: ja
#        j[numpy.ix_(self.pa_inds,self.ja_inds)] = -self.B_ps_a
#        # phi_s_c: jc
#        j[numpy.ix_(self.pc_inds,self.jc_inds)] = -self.B_ps_c
#        ###        
#
#        return j

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

    ## Helper functions for managing results data
    def get_vector( self, states ) :
        """
        Create the state vectors from the state dict
        """
        p = self.pars

        all_vars = list(self.var_inf.keys())

        y = numpy.zeros( p.num_diff_vars+p.num_algr_vars )

        for xvar in all_vars :
            y[ self.var_inf[xvar]['inds'] ] = states[xvar]

        return y


    def set_initial_states( self, init_rest_on, last_step ) :
        """
        Set the initial conditions
        """
        if init_rest_on :  # start of a new schedule and therefore all states are uniform
            yo  = self.const_init_conds()
            ydo = numpy.zeros_like(yo)
        else :
            # A step at some point after the first step, therefore the initial conditions
            # are just the states in the last step at the final time step.
            yo  = self.get_last_tstep_states( self.results_out[last_step] )
            ydo = numpy.zeros_like(yo) #self.get_dots( yo )

        return yo, ydo


    def build_results_dict( self, steps, cycs ) :
        """
        Dict data structure for the simulation results.
        Keys use the following form: 'stepX_repeatY', where X and Y are the 
        step and cycle numbers, respectively.
        """
        self.results_out = dict( [ ('step'+str(stp)+'_repeat'+str(cyc), results_object(self.pars)) for stp in range(steps) for cyc in range(cycs) ] )


    def get_last_tstep_states( self, last_dat_class ) :
        """
        Grab the data from the last time increment of the last simulation step.
        This is used for the Initial Condition setup for the next step.
        """
        last_states = self.assign_class_to_dict( last_dat_class )

        states_last_tstep = {} #dict( [ (k, []) for k in last_states.keys() ] )
        for k,v in last_states.iteritems() :
            if k in self.var_inf.keys() :
                if (type(v)==numpy.ndarray) and (len(v.shape) == 2) : # numpy 2d array
                    states_last_tstep[k] = v[:,-1]
                elif (type(v)==numpy.ndarray) and (len(v.shape) == 1) : # numpy vector
                    states_last_tstep[k] = v[-1]
                elif type(v)==list :
                    states_last_tstep[k] = v
                else :
                    print 'Unknown type in last dat class.'
                    print 'Type:',type(v)

        y_l = self.get_vector( states_last_tstep )

        return y_l

    def assign_model_results( self, states, extras, present_run ) :
        """
        Take the final solution from cn_solver and assign the values to the results
        dict.
        """
        self.assign_dict_to_class( states, self.results_out[present_run] )
        self.assign_dict_to_class( extras, self.results_out[present_run] )


    def merge_extras( self, extras ) :
        """
        Merge the list of dicts in extras into a single dict with each key being an array
        """
        NT = len(extras)
        keys = list( extras[0].keys() )

        mrgEx = dict( [ (k, []) for k in keys ] )
        for k in keys :
            v = extras[0][k]
            if (type(v)==numpy.ndarray) and len(extras[0][k].shape)==1 :
                mrgEx[k] = numpy.zeros( (len(extras[0][k]),NT) )
            elif (type(v)==numpy.float64) :
                mrgEx[k] = numpy.zeros( NT )
            elif (type(v)==float) :
                mrgEx[k] = numpy.zeros( NT )
            else :
                print type( extras[0][k] )
                print extras[0][k]
                print k
                mrgEx[k] = numpy.zeros( NT )

            
            for i in range(NT) :
                if (type(v)==numpy.ndarray) and len(extras[0][k].shape)==1 :
                    mrgEx[k][:,i] = extras[i][k]
                elif (type(v)==numpy.float64) :
                    mrgEx[k][i] = extras[i][k]
                else :
                    mrgEx[k][i] = extras[i][k]

        return mrgEx


    def assign_dict_to_class( self, dict_obj, class_obj ) :
        """
        Take the dict values and assign to results class vars.
        """
        for key, val in dict_obj.iteritems() :
            if hasattr( class_obj, key ) :
                setattr( class_obj, key, val )


    def assign_class_to_dict( self, class_obj ) :
        """
        Take the dict values and assign to results class vars.
        """
        keys = [ atr for atr in dir(class_obj) if not atr.startswith('__') ]
        dict_obj = dict( [ ( k, [] ) for k in keys ] )
        for key, val in dict_obj.iteritems() :
            if hasattr( class_obj, key ) :
                dict_obj[key] = getattr( class_obj, key )
        
        return dict_obj


    def const_init_conds( self ) :
        """
        For the first step in a schedule, setup the initial conditions, where the
        spatial states are uniform. i.e, c_s_n = csn0*numpy.zeros( p.SolidOrder_n*p.Nn )
        and csn0 is a scalar and equal to p.theta_n0*p.c_s_n_max
        """
        p = self.pars
        y0 = numpy.zeros( p.num_diff_vars+p.num_algr_vars )

        # x0
        y0[ p.ce_inds] = p.ce_0*numpy.ones( p.N, dtype='d' )
        y0[p.csa_inds] = (p.theta_a0*p.csa_max)*numpy.ones( p.Na*p.Nra, dtype='d' )
        y0[p.csc_inds] = (p.theta_c0*p.csc_max)*numpy.ones( p.Nc*p.Nrc, dtype='d' )
        y0[p.T_ind]    = deepcopy(p.T_amb)

        # z0
        y0[p.pa_inds] = p.uref_a( p.theta_a0*numpy.ones( p.Na, dtype='d' ) )
        y0[p.pc_inds] = p.uref_c( p.theta_c0*numpy.ones( p.Nc, dtype='d' ) )
        y0[p.pe_inds] = numpy.zeros( p.N, dtype='d' )
        y0[p.ja_inds] = numpy.zeros( p.Na, dtype='d' )
        y0[p.jc_inds] = numpy.zeros( p.Nc, dtype='d' )

        return y0


class results_object() :
    """
    result properties at each schedule step
    """
    def __init__(self, p) :
        """
        Define the basic properties for the results
        """
        self.c_s_a   = [] #numpy.zeros( (p.SolidOrder_n*p.Nn, NT), dtype='d' )
        self.c_s_c   = [] #numpy.zeros( (p.SolidOrder_p*p.Np, NT), dtype='d' )
        self.c_e     = [] #numpy.zeros( (p.N, NT), dtype='d' )
        self.T       = [] #numpy.zeros( NT, dtype='d' )
        self.phi_s_a = [] #numpy.zeros( (p.Na, NT), dtype='d' )
        self.phi_s_c = [] #numpy.zeros( (p.Nc, NT), dtype='d' )
        self.phi_e   = [] #numpy.zeros( (p.N, NT), dtype='d' )
        self.ja      = [] #numpy.zeros( (p.Na, NT), dtype='d' )
        self.jc      = [] #numpy.zeros( (p.Nc, NT), dtype='d' )

        self.csa_ss  = [] #numpy.zeros( (p.Nn, NT), dtype='d' )
        self.csc_ss  = [] #numpy.zeros( (p.Np, NT), dtype='d' )
        self.csa_avg = [] #numpy.zeros( (p.Nn, NT), dtype='d' )
        self.csc_avg = [] #numpy.zeros( (p.Np, NT), dtype='d' )

        self.Ua_avg = [] #numpy.zeros( NT, dtype='d' )
        self.Uc_avg = [] #numpy.zeros( NT, dtype='d' )
        self.Ua_ss  = [] #numpy.zeros( NT, dtype='d' )
        self.Uc_ss  = [] #numpy.zeros( NT, dtype='d' )
        self.eta_a  = [] #numpy.zeros( (p.Nn, NT), dtype='d' )
        self.eta_c  = [] #numpy.zeros( (p.Np, NT), dtype='d' )

        self.Va = []
        self.Vc = []
        self.css_fullx = []
        self.Uss_fullx = []
        self.phis_fullx = []
        self.eta_fullx = []
        self.j_fullx = []

        self.ke = []
        self.De = []

        self.Volt = [] #numpy.zeros( NT, dtype='d' )
        self.Cur  = [] #numpy.zeros( NT, dtype='d' )
#        self.T_dots = []
#
#        self.Ah_dod = numpy.zeros( NT, dtype='d' )
#        self.Wh_dod = numpy.zeros( NT, dtype='d' )
#
#        self.heat_loss = numpy.zeros( NT, dtype='d' )
#        self.heat_gen = numpy.zeros( NT, dtype='d' )

        self.T_amb = [] #p.T_amb*numpy.ones( NT, dtype='d' )

        self.step_time = [] #numpy.zeros( NT, dtype='d' )
        self.step_time_mins = [] #numpy.zeros( NT, dtype='d' )
        self.test_time = [] #numpy.zeros( NT, dtype='d' )
        self.test_time_mins = [] #numpy.zeros( NT, dtype='d' )

#        self.Rs = []
#        self.daeOuts = []
#        self.theta_flag = []

#        self.nLi = numpy.zeros( NT, dtype='d' )
#        self.nLidot = numpy.zeros( NT, dtype='d' )
#        self.nLi_cs = numpy.zeros( NT, dtype='d' )


#        self.newtonStats = []

class simulator() :
    
    def __init__( self, confDat ) :    
        
        self.confDat = confDat        
        
        confdir    = '/home/mk-sim-linux/Battery_TempGrad/Python/batt_simulation/battsimpy/configs/'
        
        configFile = confdir+'model_fvmP2D.conf'
        simFile    = confdir+'sim_DCR.conf'
        
        mcd = confreader.reader( configFile )
        scd = confreader.reader( simFile )
        
        conf_data = mcd.conf_data.copy()
        conf_data.update(scd.conf_data)    
    
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
        imp_mod.pars = self.p

        self.imp_mod = imp_mod        
        
        
        self.pars = self.imp_mod.p
        
#        self.model_setup = self.imp_mod.model_setup
        self.build_results_dict = self.imp_mod.build_results_dict
#        self.get_input = self.imp_mod.get_input
#        self.results_out = self.imp_mod.results_out

    
    def simulate( self, tfinal, NT, inp, init_rest_on, present_step_name, last_step_name ) :
        """
        Execute the prescribed test schedule.
        """
        p = self.imp_mod.pars

        self.results_out = self.imp_mod.results_out

#        self.model_setup()

        # Initialize the IDA simulator
        imp_sim = IDA( self.imp_mod )
        
        print 'imp_sim Vinit:', imp_sim.y[self.imp_mod.pc_inds[-1]] - imp_sim.y[self.imp_mod.pa_inds[0]]
        
        #Sets the paramters
        imp_sim.atol = 1e-5 #Default 1e-6
        imp_sim.rtol = 1e-5 #Default 1e-6
        imp_sim.suppress_alg = True #Suppres the algebraic variables on the error test

        imp_sim.display_progress = False
        imp_sim.verbosity = 50
        imp_sim.report_continuously = True
        imp_sim.time_limit = 10.

        # Update the model input current
#        inp = self.imp_mod.inp
        
        print 'Vcell check:',self.imp_mod.get_voltage(imp_sim.y)
        print 'imp_sim.t  :', imp_sim.t
        
        # Ramp up the input current
        t01, t02 = 0.01+imp_sim.t, 0.02+imp_sim.t

        print 'xa',self.imp_mod.pars.theta_a0
        print 'xc',self.imp_mod.pars.theta_c0
        print 'Ua',self.imp_mod.pars.uref_a( self.imp_mod.pars.theta_a0 )
        print 'Uc',self.imp_mod.pars.uref_c( self.imp_mod.pars.theta_c0 )
        print 'Uc-Ua', self.imp_mod.pars.uref_c( self.imp_mod.pars.theta_c0 ) - self.imp_mod.pars.uref_a( self.imp_mod.pars.theta_a0 )   
        self.imp_mod.V_init = self.imp_mod.get_voltage(imp_sim.y)
        print 'impmod Vinit', self.imp_mod.V_init
        
        print 'Tamb', self.imp_mod.T_amb
        print 'To', self.imp_mod.y0[self.imp_mod.T_ind]        
        
        self.imp_mod.set_iapp( inp/10. )
        imp_sim.make_consistent('IDA_YA_YDP_INIT')
        ta, ya, yda = imp_sim.simulate(t01,2) 

        self.imp_mod.set_iapp( inp/2. )
        imp_sim.make_consistent('IDA_YA_YDP_INIT')
        tb, yb, ydb = imp_sim.simulate(t02,2) 

        ti = imp_sim.t
        delta_t = tfinal/NT

        V_cell = self.imp_mod.get_voltage( imp_sim.y )
        ce_now = imp_sim.y[p.ce_inds]
        print 'V_cell prior to time loop:', V_cell

        print 'present_step_name:', present_step_name
        print 'self.inp:', self.inp
        
        self.imp_mod.set_iapp( inp )
        imp_sim.make_consistent('IDA_YA_YDP_INIT')
        sim_stopped = 0

        dV_tol  = 0.002
        delta_t = 0.1
        refined_dt = 0

        V_out = []
        t_out = []

        y_out = []
        yd_out = []

        xa_bar = []
        xc_bar = []
        Ua_bar = []
        Uc_bar = []

        csa_avg_out = []
        csc_avg_out = []
        Ua_avg_out = []
        Uc_avg_out = []

        csa_ss_out = []
        csc_ss_out = []
        Ua_ss_out = []
        Uc_ss_out = []

        eta_a_out = []
        eta_c_out = []
        eta_full_x = []

        j_full_x = []
        Uss_full_x = []
        css_full_x = []
        phis_full_x = []

        ke_out = []
        De_out = []

        # Perform the simulation
        it = 0
        while V_cell > p.volt_min and V_cell < p.volt_max and max(ce_now)<max(p.ce_lims) and min(ce_now)>min(p.ce_lims) and not sim_stopped and imp_sim.t<=tfinal :

            # delta_t adapt based on dV/dt
            if it > 2 and not refined_dt :
                dV = abs(V_out[-1] - V_out[-2])
                delta_t = dV_tol/dV * delta_t

            # Final delta_t alignment
            if (imp_sim.t + delta_t) > tfinal :
                delta_t = tfinal - imp_sim.t + .00001

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
    #                    ti  = [t_out[it-1],t_out[it-1]]
    #                    yi  = numpy.array([ y_out[it-2], y_out[it-1] ])
    #                    ydi = numpy.array([ yd_out[it-2], yd_out[it-1] ])

                    sim_stopped = 1

                    print 'Sim stopped due time integration failure.'

            t_out.append( imp_sim.t )
            y_out.append( imp_sim.y )
            yd_out.append( imp_sim.yd )

            csa_now = imp_sim.y[p.csa_inds]
            csc_now = imp_sim.y[p.csc_inds]

            csa_avg_now = self.imp_mod.C_cs_a_avg.dot(csa_now)
            csc_avg_now = self.imp_mod.C_cs_c_avg.dot(csc_now)

            Ua_avg_now = p.uref_a( csa_avg_now/p.csa_max )
            Uc_avg_now = p.uref_c( csc_avg_now/p.csc_max )

            csa_avg_out.append(csa_avg_now)
            csc_avg_out.append(csc_avg_now)
            Ua_avg_out.append(Ua_avg_now)
            Uc_avg_out.append(Uc_avg_now)

            csa_mean = numpy.mean( csa_avg_now )
            csc_mean = numpy.mean( csc_avg_now )
            xa_mean  = csa_mean/p.csa_max
            xc_mean  = csc_mean/p.csc_max
            Uam = p.uref_a( xa_mean )
            Ucm = p.uref_c( xc_mean )

            xa_bar.append(xa_mean)
            xc_bar.append(xc_mean)
            Ua_bar.append(Uam)
            Uc_bar.append(Ucm)

            V_cell = self.imp_mod.get_voltage(imp_sim.y)

            V_out.append( V_cell )

            ce_now = imp_sim.y[p.ce_inds]

            ja_now = imp_sim.y[p.ja_inds]
            jc_now = imp_sim.y[p.jc_inds]
            pa_now = imp_sim.y[p.pa_inds]
            pc_now = imp_sim.y[p.pc_inds]
#            pe_now = imp_sim.y[p.pe_inds]

#            T_now  = imp_sim.y[p.T_ind]

            csa_ss, csc_ss = self.imp_mod.get_css()
            eta_a, eta_c, Uref_a_ss, Uref_c_ss = self.imp_mod.get_eta_uref()

            eta_a_out.append(eta_a)
            eta_c_out.append(eta_c)

            eta_full_x.append(  numpy.concatenate( [eta_a ,    numpy.zeros(p.Ns), eta_c    ] ) )
            j_full_x.append(    numpy.concatenate( [ja_now,    numpy.zeros(p.Ns), jc_now   ] ) )
            Uss_full_x.append(  numpy.concatenate( [Uref_a_ss, numpy.zeros(p.Ns), Uref_c_ss] ) )
            phis_full_x.append( numpy.concatenate( [pa_now,    numpy.zeros(p.Ns), pc_now   ] ) )
            css_full_x.append(  numpy.concatenate( [csa_ss,    numpy.zeros(p.Ns), csc_ss   ] ) )

            csa_ss_out.append(csa_ss)
            csc_ss_out.append(csc_ss)

            ke_mid = self.imp_mod.kapp_ce( imp_sim.y, mid_on=1, eps_off=1 )
            De_mid = self.imp_mod.Diff_ce( imp_sim.y, mid_on=1, eps_off=1 )

            ke_out.append( ke_mid )
            De_out.append( De_mid )

            print 'time:',round(imp_sim.t,3), ' |  Voltage:', round(V_cell,3), ' |  '+str(round(imp_sim.t/tfinal*100.,1))+'% complete'

            if V_cell < p.volt_min :
                print '\n','Vcut stopped simulation.'
            elif max(ce_now)>max(p.ce_lims) :
                print '\n','ce max stopped simulation.'
            elif min(ce_now)<min(p.ce_lims) :
                print '\n','ce min stopped simulation.'

            it+=1

        y1  = numpy.array(y_out)
#        yd1 = numpy.array(yd_out)

        states = {}
        states['c_s_a'] = y1[:,p.csa_inds]
        states['c_s_c'] = y1[:,p.csc_inds]
        states['c_e']   = y1[:,p.ce_inds]
        states['T']     = y1[:,p.T_ind]

        states['phi_e']   = y1[:,p.pe_inds]
        states['phi_s_a'] = y1[:,p.pa_inds]
        states['phi_s_c'] = y1[:,p.pc_inds]

        states['ja'] = y1[:,p.ja_inds]
        states['jc'] = y1[:,p.jc_inds]

        pa_cc = states['phi_s_a'][:, 0]
        pc_cc = states['phi_s_c'][:,-1]
        pe_midsep = states['phi_e'][:,int(p.Na+(p.Ns/2.))]

        Va = pa_cc - pe_midsep
        Vc = pc_cc - pe_midsep

        mergExtr = {}
        mergExtr['Volt'] = numpy.array(V_out)
        mergExtr['Va'] = Va
        mergExtr['Vc'] = Vc
        mergExtr['step_time'] = numpy.array( t_out ) - t_out[0]
        mergExtr['step_time_mins'] = mergExtr['step_time']/60.

        mergExtr['test_time'] = numpy.array( t_out )
        mergExtr['test_time_mins'] = mergExtr['test_time']/60.
     
        mergExtr[ 'css_fullx'] = numpy.array(  css_full_x )
        mergExtr[ 'Uss_fullx'] = numpy.array(  Uss_full_x )
        mergExtr['phis_fullx'] = numpy.array( phis_full_x )
        mergExtr[ 'eta_fullx'] = numpy.array(  eta_full_x )
        mergExtr[   'j_fullx'] = numpy.array(    j_full_x )

        mergExtr['eta_a'] = numpy.array( eta_a_out )
        mergExtr['eta_c'] = numpy.array( eta_c_out )

        mergExtr['csa_avg'] = numpy.array( csa_avg_out )
        mergExtr['csc_avg'] = numpy.array( csc_avg_out )
        mergExtr['csa_ss']  = numpy.array( csa_ss_out )
        mergExtr['csc_ss']  = numpy.array( csc_ss_out )

        mergExtr['Ua_avg'] = numpy.array( Ua_avg_out )
        mergExtr['Uc_avg'] = numpy.array( Uc_avg_out )
        mergExtr['Ua_ss']  = numpy.array( Ua_ss_out )
        mergExtr['Uc_ss']  = numpy.array( Uc_ss_out )

        mergExtr['ke'] = numpy.array( ke_out )
        mergExtr['De'] = numpy.array( De_out )
        
        self.t_end_now = imp_sim.t

        self.imp_mod.assign_model_results( states, mergExtr, present_step_name )

        return self.imp_mod.results[present_step_name]


#def simulate( model_obj, t_end_last, tfinal, NT, inp, init_rest_on, present_step_name, last_step_name ) :
#    """
#    Execute the prescribed test schedule.
#    """
#    p = model_obj.pars

#    model_obj.inp = inp

#    volt_lim_on = 0

#    # Initialize the IDA simulator
#    imp_sim = IDA(model_obj)
#    #Sets the paramters
#    imp_sim.atol = 1e-5 #Default 1e-6
#    imp_sim.rtol = 1e-5 #Default 1e-6
#    imp_sim.suppress_alg = True #Suppres the algebraic variables on the error test

#    imp_sim.display_progress = False
#    imp_sim.verbosity = 50
#    imp_sim.report_continuously = True
#    imp_sim.time_limit = 10.

##    imp_sim.usejac = True
#    
#    print 'vcell check:',model_obj.get_voltage( imp_sim.y )        
#    print 'imp_sim.t', imp_sim.t
#    
#    # Ramp up the input current
##        t01, t02 = 0.01, 0.02
#    t01, t02 = 0.01+t_end_last, 0.02+t_end_last

#    model_obj.set_iapp( model_obj.inp/10. )
#    imp_sim.make_consistent('IDA_YA_YDP_INIT')
#    ta, ya, yda = imp_sim.simulate(t01,2) 

#    model_obj.set_iapp( model_obj.inp/2. )
#    imp_sim.make_consistent('IDA_YA_YDP_INIT')
#    tb, yb, ydb = imp_sim.simulate(t02,2) 

#    ti = imp_sim.t
#    delta_t = tfinal/NT

#    V_cell = model_obj.get_voltage( imp_sim.y )
#    ce_now = imp_sim.y[p.ce_inds]
#    print 'V_cell prior to time loop:', V_cell

#    print present_step_name
#    print model_obj.inp
#    
#    model_obj.set_iapp( model_obj.inp )
#    imp_sim.make_consistent('IDA_YA_YDP_INIT')
#    sim_stopped = 0

#    dV_tol  = 0.002
#    delta_t = 0.1
#    refined_dt = 0

#    V_out = []
#    t_out = []

#    y_out = []
#    yd_out = []

#    xa_bar = []
#    xc_bar = []
#    Ua_bar = []
#    Uc_bar = []

#    csa_avg_out = []
#    csc_avg_out = []
#    Ua_avg_out = []
#    Uc_avg_out = []

#    csa_ss_out = []
#    csc_ss_out = []
#    Ua_ss_out = []
#    Uc_ss_out = []

#    eta_a_out = []
#    eta_c_out = []
#    eta_full_x = []

#    j_full_x = []
#    Uss_full_x = []
#    css_full_x = []
#    phis_full_x = []

#    ke_out = []
#    De_out = []

#    # Perform the simulation
#    it = 0
#    while V_cell > p.volt_min and V_cell < p.volt_max and max(ce_now)<max(p.ce_lims) and min(ce_now)>min(p.ce_lims) and not sim_stopped and imp_sim.t<=tfinal :

#        # delta_t adapt based on dV/dt
#        if it > 2 and not refined_dt :
#            dV = abs(V_out[-1] - V_out[-2])
#            delta_t = dV_tol/dV * delta_t

#        # Final delta_t alignment
#        if (imp_sim.t + delta_t) > tfinal :
#            delta_t = tfinal - imp_sim.t + .00001

#        try :
#            ti, yi, ydi = imp_sim.simulate(imp_sim.t+delta_t,2)

#        except :
#            try :
#                if imp_sim.t > 0.8*tfinal :
#                    delta_t = delta_t*.1
#                    refined_dt = 1
#                ti, yi, ydi = imp_sim.simulate(imp_sim.t+delta_t*.01,3)
#                print '*** ran with refined delta_t ***'

#            except :
##                    ti  = [t_out[it-1],t_out[it-1]]
##                    yi  = numpy.array([ y_out[it-2], y_out[it-1] ])
##                    ydi = numpy.array([ yd_out[it-2], yd_out[it-1] ])

#                sim_stopped = 1

#                print 'Sim stopped due time integration failure.'

#        t_out.append( imp_sim.t )
#        y_out.append( imp_sim.y )
#        yd_out.append( imp_sim.yd )

#        csa_now = imp_sim.y[p.csa_inds]
#        csc_now = imp_sim.y[p.csc_inds]

#        csa_avg_now = model_obj.C_cs_a_avg.dot(csa_now)
#        csc_avg_now = model_obj.C_cs_c_avg.dot(csc_now)

#        Ua_avg_now = p.uref_a( csa_avg_now/p.csa_max )
#        Uc_avg_now = p.uref_c( csc_avg_now/p.csc_max )

#        csa_avg_out.append(csa_avg_now)
#        csc_avg_out.append(csc_avg_now)
#        Ua_avg_out.append(Ua_avg_now)
#        Uc_avg_out.append(Uc_avg_now)

#        csa_mean = numpy.mean( csa_avg_now )
#        csc_mean = numpy.mean( csc_avg_now )
#        xa_mean  = csa_mean/p.csa_max
#        xc_mean  = csc_mean/p.csc_max
#        Uam = p.uref_a( xa_mean )
#        Ucm = p.uref_c( xc_mean )

#        xa_bar.append(xa_mean)
#        xc_bar.append(xc_mean)
#        Ua_bar.append(Uam)
#        Uc_bar.append(Ucm)

#        V_cell = model_obj.get_voltage( imp_sim.y )

#        V_out.append( V_cell )

#        ce_now = imp_sim.y[p.ce_inds]

#        ja_now = imp_sim.y[p.ja_inds]
#        jc_now = imp_sim.y[p.jc_inds]
#        pa_now = imp_sim.y[p.pa_inds]
#        pc_now = imp_sim.y[p.pc_inds]
#        pe_now = imp_sim.y[p.pe_inds]

#        T_now  = imp_sim.y[p.T_ind]

#        eta_a, eta_c, Uref_a_ss, Uref_c_ss, csa_ss, csc_ss = model_obj.get_eta_uref( csa_now, csc_now, ja_now, jc_now, pa_now, pc_now, pe_now )

#        eta_a_out.append(eta_a)
#        eta_c_out.append(eta_c)

#        eta_full_x.append(  numpy.concatenate( [eta_a ,    numpy.zeros(p.Ns), eta_c    ] ) )
#        j_full_x.append(    numpy.concatenate( [ja_now,    numpy.zeros(p.Ns), jc_now   ] ) )
#        Uss_full_x.append(  numpy.concatenate( [Uref_a_ss, numpy.zeros(p.Ns), Uref_c_ss] ) )
#        phis_full_x.append( numpy.concatenate( [pa_now,    numpy.zeros(p.Ns), pc_now   ] ) )
#        css_full_x.append(  numpy.concatenate( [csa_ss,    numpy.zeros(p.Ns), csc_ss   ] ) )

#        csa_ss_out.append(csa_ss)
#        csc_ss_out.append(csc_ss)

#        ke_mid = model_obj.kapp_ce( ce_now, T_now, mid_on=1, eps_off=1 )
#        De_mid = model_obj.Diff_ce( ce_now, T_now, mid_on=1, eps_off=1 )

#        ke_out.append( ke_mid )
#        De_out.append( De_mid )

#        print 'time:',round(imp_sim.t,3), ' |  Voltage:', round(V_cell,3), ' |  '+str(round(imp_sim.t/tfinal*100.,1))+'% complete'

#        if V_cell < p.volt_min :
#            print '\n','Vcut stopped simulation.'
#        elif max(ce_now)>max(p.ce_lims) :
#            print '\n','ce max stopped simulation.'
#        elif min(ce_now)<min(p.ce_lims) :
#            print '\n','ce min stopped simulation.'

#        it+=1

#    y1  = numpy.array(y_out)
#    yd1 = numpy.array(yd_out)

#    states = {}
#    states['c_s_a'] = y1[:,p.csa_inds]
#    states['c_s_c'] = y1[:,p.csc_inds]
#    states['c_e']   = y1[:,p.ce_inds]
#    states['T']     = y1[:,p.T_ind]

#    states['phi_e']   = y1[:,p.pe_inds]
#    states['phi_s_a'] = y1[:,p.pa_inds]
#    states['phi_s_c'] = y1[:,p.pc_inds]

#    states['ja'] = y1[:,p.ja_inds]
#    states['jc'] = y1[:,p.jc_inds]

#    pa_cc = states['phi_s_a'][:, 0]
#    pc_cc = states['phi_s_c'][:,-1]
#    pe_midsep = states['phi_e'][:,int(p.Na+(p.Ns/2.))]

#    Va = pa_cc - pe_midsep
#    Vc = pc_cc - pe_midsep

#    mergExtr = {}
#    mergExtr['Volt'] = numpy.array(V_out)
#    mergExtr['Va'] = Va
#    mergExtr['Vc'] = Vc
#    mergExtr['step_time'] = numpy.array( t_out )
#    mergExtr['step_time_mins'] = mergExtr['step_time']/60.

#    mergExtr['test_time'] = numpy.array( t_out ) + model_obj.t_end_now
#    mergExtr['test_time_mins'] = mergExtr['test_time']/60.
# 
#    mergExtr[ 'css_fullx'] = numpy.array(  css_full_x )
#    mergExtr[ 'Uss_fullx'] = numpy.array(  Uss_full_x )
#    mergExtr['phis_fullx'] = numpy.array( phis_full_x )
#    mergExtr[ 'eta_fullx'] = numpy.array(  eta_full_x )
#    mergExtr[   'j_fullx'] = numpy.array(    j_full_x )

#    mergExtr['eta_a'] = numpy.array( eta_a_out )
#    mergExtr['eta_c'] = numpy.array( eta_c_out )

#    mergExtr['csa_avg'] = numpy.array( csa_avg_out )
#    mergExtr['csc_avg'] = numpy.array( csc_avg_out )
#    mergExtr['csa_ss']  = numpy.array( csa_ss_out )
#    mergExtr['csc_ss']  = numpy.array( csc_ss_out )

#    mergExtr['Ua_avg'] = numpy.array( Ua_avg_out )
#    mergExtr['Uc_avg'] = numpy.array( Uc_avg_out )
#    mergExtr['Ua_ss']  = numpy.array( Ua_ss_out )
#    mergExtr['Uc_ss']  = numpy.array( Uc_ss_out )

#    mergExtr['ke'] = numpy.array( ke_out )
#    mergExtr['De'] = numpy.array( De_out )
#    
#    model_obj.t_end_now = imp_sim.t+model_obj.t_end_now

#    model_obj.assign_model_results_mk( states, mergExtr, present_step_name )

#    return model_obj.results[present_step_name]


#def plotresults( self ) :
#    """
#    Create a few key figures for create plots of results
#    """

