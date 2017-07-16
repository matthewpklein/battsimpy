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
import scipy.sparse as sparse
import math
from copy import deepcopy


def ButterworthFilter( x, y, ff=0.2 ) :
    b, a = butter(1, ff)
    fl = filtfilt( b, a, y )
    return fl

def get_smooth_Uref_data( Ua_path, Uc_path, ffa=0.4, ffc=0.2 ) :
    """
    Smooth the Uref data to aid in improving numerical stability.
    This should be verified by the user to ensure it is not changing the original
    Uref data beyond a tolerable amount (defined by the user).
    A linear interpolator class is output for Uref and dUref_dx for both anode 
    and cathode.
    """
    ## Load the data files
    uref_a_map = numpy.loadtxt( Ua_path, delimiter=',' )
    uref_c_map = numpy.loadtxt( bsp_dir + '/data/Model_v1/Model_Pars/solid/thermodynamics/uref_cathode_bigx.csv', delimiter=',' )

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
    Ua_butter = ButterworthFilter( xa, uref_a_map[:,1], ff=ffa )
    Uc_butter = ButterworthFilter( xc, uref_c_map[:,1], ff=ffc )

    ## Create the interpolators
    Ua_intp = scipy.interpolate.interp1d( xa, Ua_butter, kind='linear' )
    Uc_intp = scipy.interpolate.interp1d( xc, Uc_butter, kind='linear' )

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

#    plt.plot( xa, self.uref_a_interp(xa), label='Ua interp lin' )
#    plt.plot( xc, self.uref_c_interp(xc), label='Uc interp lin' )
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

class MyProblem( Implicit_Problem ) :

    def __init__(self, Na, Ns, Nc, Nra, Nrc, X, Ra, Rc, Ac, bsp_dir, y0, yd0, name ) :

        Implicit_Problem.__init__(self,y0=y0,yd0=yd0,name=name)

        self.T  = 298.15 # Cell temperature, [K]
        self.Ac = Ac # Cell coated area, [m^2]

        # Control volumes and node points (mid node points and edge node points)
        self.Ns = Ns
        self.Na = Na
        self.Nc = Nc

        self.N = Na + Ns + Nc
        self.X = X

        self.x_e  = numpy.linspace( 0.0, X, N+1 )
        self.x_m  = numpy.array( [ 0.5*(self.x_e[i+1]+self.x_e[i]) for i in range(N) ], dtype='d'  )
        self.vols = numpy.array( [ (self.x_e[i+1] - self.x_e[i]) for i in range(N)], dtype='d' )

        # Radial mesh
        self.Nra = Nra
        self.Nrc = Nrc

        k=0.8
        self.r_e_a = nonlinspace( Ra, k, Nra )
        self.r_m_a  = numpy.array( [ 0.5*(self.r_e_a[i+1]+self.r_e_a[i]) for i in range(Nra-1) ], dtype='d'  )
        self.r_e_c = nonlinspace( Rc, k, Nrc )
        self.r_m_c  = numpy.array( [ 0.5*(self.r_e_c[i+1]+self.r_e_c[i]) for i in range(Nrc-1) ], dtype='d'  )
        self.vols_ra_e = numpy.array( [1/3.*(self.r_m_a[0]**3)] + [ 1/3.*(self.r_m_a[i+1] **3 - self.r_m_a[i] **3) for i in range(Nra-2)] + [1/3.*(self.r_e_a[-1]**3 - self.r_m_a[-1]**3)], dtype='d' )
        self.vols_rc_e = numpy.array( [1/3.*(self.r_m_c[0]**3)] + [ 1/3.*(self.r_m_c[i+1] **3 - self.r_m_c[i] **3) for i in range(Nrc-2)] + [1/3.*(self.r_e_c[-1]**3 - self.r_m_c[-1]**3)], dtype='d' )
        self.vols_ra_m = numpy.array( [ 1/3.*(self.r_e_a[i+1]**3 - self.r_e_a[i]**3) for i in range(Nra-1)], dtype='d' )
        self.vols_rc_m = numpy.array( [ 1/3.*(self.r_e_c[i+1]**3 - self.r_e_c[i]**3) for i in range(Nrc-1)], dtype='d' )

        # Useful sub-meshes for the phi_s functions
        self.x_m_a = self.x_m[:Na]
        self.x_m_c = self.x_m[-Nc:]
        self.x_e_a = self.x_e[:Na+1]
        self.x_e_c = self.x_e[-Nc-1:]

        self.vols_a = self.vols[:Na]
        self.vols_c = self.vols[-Nc:]

        self.num_diff_vars = self.N + self.Nra*self.Na + self.Nrc*self.Nc
        self.num_algr_vars = self.N + self.Na+self.Nc

        # Volume fraction vectors and matrices for effective parameters
        self.La, self.Ls, self.Lc = self.Na*X/self.N, self.Ns*X/self.N, self.Nc*X/self.N
        self.Na, self.Ns, self.Nc = Na, Ns, Nc
        eps_a = 0.3
        eps_s = 0.5
        eps_c = 0.25
        ba, bs, bc = 0.8, 0.5, 0.5

        eps_a_vec = [ eps_a for i in range(Na) ] # list( eps_a + eps_a/2.*numpy.sin(numpy.linspace(0.,Na/4,Na)) ) # list(eps_a + eps_a*numpy.random.randn(Na)/5.) #
        eps_s_vec = [ eps_s for i in range(Ns) ]
        eps_c_vec = [ eps_c for i in range(Nc) ] # list( eps_c + eps_c/2.*numpy.sin(numpy.linspace(0.,Nc/4,Nc)) ) # list(eps_c + eps_c*numpy.random.randn(Nc)/5.) #

        self.eps_m   = numpy.array( eps_a_vec + eps_s_vec + eps_c_vec, dtype='d' )
        self.k_m     = 1./self.eps_m
        self.eps_mb  = numpy.array( [ ea**ba for ea in eps_a_vec ] + [ es**bs for es in eps_s_vec ] + [ ec**bc for ec in eps_c_vec ], dtype='d' )
        self.eps_eff = numpy.array( [ ea**(1.+ba) for ea in eps_a_vec ] + [ es**(1.+bs) for es in eps_s_vec ] + [ ec**(1.+bc) for ec in eps_c_vec ], dtype='d' )

        self.eps_a_eff = self.eps_eff[:Na]
        self.eps_c_eff = self.eps_eff[-Nc:]

        self.K_m = numpy.diag( self.k_m )

        t_plus = 0.4
        F = 96485.0

        self.t_plus = t_plus
        self.F = F
        self.R_gas = 8.314

        self.Rp_a = Ra
        self.Rp_c = Rc

        as_a = 3.*numpy.array(eps_a_vec, dtype='d')/self.Rp_a
        as_c = 3.*numpy.array(eps_c_vec, dtype='d')/self.Rp_c
        self.as_a = as_a
        self.as_c = as_c

        self.as_a_mean = 1./self.La*sum( [ asa*v for asa,v in zip(as_a, self.vols[:Na]) ] )
        self.as_c_mean = 1./self.Lc*sum( [ asc*v for asc,v in zip(as_c, self.vols[-Nc:]) ] )

        print 'asa diff', self.as_a_mean - as_a[0]
        print 'asc diff', self.as_c_mean - as_c[0]

        # Electrolyte constant B_ce matrix
        Ba = [ (1.-t_plus)*asa/ea for ea, asa in zip(eps_a_vec,as_a) ]
        Bs = [  0.0                for i in range(Ns) ]
        Bc = [ (1.-t_plus)*asc/ec for ec, asc in zip(eps_c_vec,as_c) ]

        self.B_ce = numpy.diag( numpy.array(Ba+Bs+Bc, dtype='d') )

        Bap = [ asa*F for asa in as_a  ]
        Bsp = [   0.0 for i   in range(Ns) ]
        Bcp = [ asc*F for asc in as_c  ]

        self.B2_pe = numpy.diag( numpy.array(Bap+Bsp+Bcp, dtype='d') )

        # Solid phase parameters and j vector matrices
        self.sig_a = 100. # [S/m]
        self.sig_c = 100. # [S/m]

        self.sig_a_eff = self.sig_a * self.eps_a_eff
        self.sig_c_eff = self.sig_c * self.eps_c_eff

        self.A_ps_a = flux_mat_builder( self.Na, self.x_m_a, numpy.ones_like(self.vols_a), self.sig_a_eff )
        self.A_ps_c = flux_mat_builder( self.Nc, self.x_m_c, numpy.ones_like(self.vols_c), self.sig_c_eff )

        # Grounding form for BCs (was only needed during testing, before BVK was incorporated for coupling
        self.A_ps_a[-1,-1] = 2*self.A_ps_a[-1,-1]
        self.A_ps_c[ 0, 0] = 2*self.A_ps_c[ 0, 0]

        Baps = numpy.array( [ asa*F*dxa for asa,dxa in zip(as_a, self.vols_a) ], dtype='d' )
        Bcps = numpy.array( [ asc*F*dxc for asc,dxc in zip(as_c, self.vols_c) ], dtype='d' )

        self.B_ps_a = numpy.diag( Baps )
        self.B_ps_c = numpy.diag( Bcps )

        self.B2_ps_a = numpy.zeros( self.Na, dtype='d' )
        self.B2_ps_a[ 0] = -1.
        self.B2_ps_c = numpy.zeros( self.Nc, dtype='d' )
        self.B2_ps_c[-1] = -1.

        # Solid phase diffusion model
        Dsa = 1e-12
        Dsc = 1e-14
        self.Dsa = Dsa
        self.Dsc = Dsc

        self.csa_max = 30555.0 # [mol/m^3]
        self.csc_max = 51554.0 # [mol/m^3]

        # Two parameter Solid phase diffusion model
#        self.B_cs_a = numpy.diag( numpy.array( [-3.0/self.Rp_a for i in range(Na)], dtype='d' ) ) 
#        self.B_cs_c = numpy.diag( numpy.array( [-3.0/self.Rp_c for i in range(Nc)], dtype='d' ) ) 

#        self.C_cs_a = numpy.eye(Na)
#        self.C_cs_c = numpy.eye(Nc)

#        self.D_cs_a = numpy.diag( numpy.array( [-self.Rp_a/Dsa/5.0 for i in range(Na)], dtype='d' ) ) 
#        self.D_cs_c = numpy.diag( numpy.array( [-self.Rp_c/Dsc/5.0 for i in range(Nc)], dtype='d' ) ) 

        # 1D spherical diffusion model
        self.A_csa_single = self.build_Ac_mat( Nra, Dsa*numpy.ones_like(self.r_m_a), self.r_m_a, self.r_e_a, self.vols_ra_e )
        self.A_csc_single = self.build_Ac_mat( Nrc, Dsc*numpy.ones_like(self.r_m_c), self.r_m_c, self.r_e_c, self.vols_rc_e )

#        b = self.A_csa_single.reshape(1,Nra,Nra).repeat(Na,axis=0)
        b = [self.A_csa_single]*Na
        self.A_cs_a = scipy.linalg.block_diag( *b )
        b = [self.A_csc_single]*Nc
        self.A_cs_c = scipy.linalg.block_diag( *b )

        B_csa_single = numpy.array( [ 0. for i in range(Nra) ], dtype='d' )
        B_csa_single[-1] = -1.*self.r_e_a[-1]**2
        A1 = self.build_Mc_A1mat( self.vols_ra_e )
        self.B_csa_single = A1.dot(B_csa_single)

        B_csc_single = numpy.array( [ 0. for i in range(Nrc) ], dtype='d' )
        B_csc_single[-1] = -1.*self.r_e_c[-1]**2
        A1 = self.build_Mc_A1mat( self.vols_rc_e )
        self.B_csc_single = A1.dot(B_csc_single)

        b = [self.B_csa_single]*Na
        self.B_cs_a = scipy.linalg.block_diag( *b ).T
        b = [self.B_csc_single]*Nc
        self.B_cs_c = scipy.linalg.block_diag( *b ).T

        self.D_cs_a = scipy.linalg.block_diag( *[[0.0 for i in range(self.Nra-1)]+[1.0]]*Na )
        self.D_cs_c = scipy.linalg.block_diag( *[[0.0 for i in range(self.Nrc-1)]+[1.0]]*Nc )

        # OCV
        Ua_path = bsp_dir+'data/Model_v1/Model_Pars/solid/thermodynamics/uref_anode_bigx.csv'
        Uc_path = bsp_dir+'data/Model_v1/Model_Pars/solid/thermodynamics/uref_cathode_bigx.csv'

        self.uref_a, self.uref_c, self.duref_a, self.duref_c = get_smooth_Uref_data( Ua_path, Uc_path, ffa=0.4, ffc=0.2 )

        # Reaction kinetics parameters
        self.io_a = 1.0 # [A/m^2]
        self.io_c = 1.0 # [A/m^2]

        ## System indices
        # Differential vars
        self.ce_inds   = range( self.N )
        self.ce_inds_r = numpy.reshape( self.ce_inds, [len(self.ce_inds),1] )
        self.ce_inds_c = numpy.reshape( self.ce_inds, [1,len(self.ce_inds)] )

        self.csa_inds = range( self.N, self.N + (self.Na*self.Nra) )
        self.csa_inds_r = numpy.reshape( self.csa_inds, [len(self.csa_inds),1] )
        self.csa_inds_c = numpy.reshape( self.csa_inds, [1,len(self.csa_inds)] )

        self.csc_inds = range( self.N + (self.Na*self.Nra), self.N + (self.Na*self.Nra) + (self.Nc*self.Nrc) )
        self.csc_inds_r = numpy.reshape( self.csc_inds, [len(self.csc_inds),1] )
        self.csc_inds_c = numpy.reshape( self.csc_inds, [1,len(self.csc_inds)] )

        # Algebraic vars
        c_end = self.N + (self.Na*self.Nra) + (self.Nc*self.Nrc)
        self.pe_inds   = range( c_end, c_end +self.N )
        self.pe_inds_r = numpy.reshape( self.pe_inds, [len(self.pe_inds),1] )
        self.pe_inds_c = numpy.reshape( self.pe_inds, [1,len(self.pe_inds)] )

        self.pe_a_inds = range( c_end, c_end +self.Na )
        self.pe_a_inds_r = numpy.reshape( self.pe_a_inds, [len(self.pe_a_inds),1] )
        self.pe_a_inds_c = numpy.reshape( self.pe_a_inds, [1,len(self.pe_a_inds)] )

        self.pe_c_inds = range( c_end+self.Na+self.Ns, c_end+self.Na+self.Ns +self.Nc )
        self.pe_c_inds_r = numpy.reshape( self.pe_c_inds, [len(self.pe_c_inds),1] )
        self.pe_c_inds_c = numpy.reshape( self.pe_c_inds, [1,len(self.pe_c_inds)] )

        self.pa_inds = range( c_end+self.N, c_end+self.N +self.Na )
        self.pa_inds_r = numpy.reshape( self.pa_inds, [len(self.pa_inds),1] )
        self.pa_inds_c = numpy.reshape( self.pa_inds, [1,len(self.pa_inds)] )

        self.pc_inds = range( c_end+self.N+self.Na, c_end+self.N+self.Na + self.Nc )
        self.pc_inds_r = numpy.reshape( self.pc_inds, [len(self.pc_inds),1] )
        self.pc_inds_c = numpy.reshape( self.pc_inds, [1,len(self.pc_inds)] )
        

        # second set for manual jac version
        c_end = 0
        self.pe_inds2  = range( c_end, c_end +self.N )
        self.pe_inds_r2 = numpy.reshape( self.pe_inds2, [len(self.pe_inds2),1] )
        self.pe_inds_c2 = numpy.reshape( self.pe_inds2, [1,len(self.pe_inds2)] )

        self.pe_a_inds2 = range( c_end, c_end +self.Na )
        self.pe_a_inds_r2 = numpy.reshape( self.pe_a_inds2, [len(self.pe_a_inds2),1] )
        self.pe_a_inds_c2 = numpy.reshape( self.pe_a_inds2, [1,len(self.pe_a_inds2)] )

        self.pe_c_inds2 = range( c_end+self.Na+self.Ns, c_end+self.Na+self.Ns +self.Nc )
        self.pe_c_inds_r2 = numpy.reshape( self.pe_c_inds2, [len(self.pe_c_inds2),1] )
        self.pe_c_inds_c2 = numpy.reshape( self.pe_c_inds2, [1,len(self.pe_c_inds2)] )

        self.pa_inds2 = range( c_end+self.N, c_end+self.N +self.Na )
        self.pa_inds_r2 = numpy.reshape( self.pa_inds2, [len(self.pa_inds2),1] )
        self.pa_inds_c2 = numpy.reshape( self.pa_inds2, [1,len(self.pa_inds2)] )

        self.pc_inds2 = range( c_end+self.N+self.Na, c_end+self.N+self.Na + self.Nc )
        self.pc_inds_r2 = numpy.reshape( self.pc_inds2, [len(self.pc_inds2),1] )
        self.pc_inds_c2 = numpy.reshape( self.pc_inds2, [1,len(self.pc_inds2)] )

    def set_iapp( self, I_app ) :
        i_app = I_app / self.Ac
        self.i_app = i_app        
        
        j_in_a  =  i_app / ( self.La*self.as_a_mean*self.F ) 
        j_in_c  = -i_app / ( self.Lc*self.as_c_mean*self.F ) 

        # Set the input j
        ja = [ j_in_a for i in range(self.Na) ]
        js = [  0.0  for i in range(self.Ns) ]
        jc = [ j_in_c for i in range(self.Nc) ]

        self.j = numpy.array( ja+js+jc, dtype='d'  )

        self.j_a = numpy.array( ja, dtype='d' )
        self.j_c = numpy.array( jc, dtype='d' )

    # cs mats
    def build_Mc_A1mat( self, V ) :
        
        M1 = numpy.zeros( [len(V), len(V)] )
        M2 = numpy.diag( V )

        M1[ 0,[0 , 1]] = [3/8., 1/8.]
        M1[-1,[-2,-1]] = [1/8., 3/8.]

        for i in range(1,len(V)-1) :
            M1[i,[i-1,i,i+1]] = [ 1/8., 6/8., 1/8. ]

        Mc = M1.dot(M2)

        A1 = numpy.linalg.inv( Mc )

        return A1


    def build_A2_mat( self, N, D, r_mid, r_edge, vols ) :
        
        A2 = numpy.zeros( [N,N] )

        for i in range(1,N-1) :
            A2[i,i-1] =  (D[i-1]*(r_mid[i-1]**2)) / (r_edge[i  ] - r_edge[i-1])
            A2[i,i  ] = -(D[i-1]*(r_mid[i-1]**2)) / (r_edge[i  ] - r_edge[i-1]) - (D[i]*(r_mid[i]**2)) / (r_edge[i+1] - r_edge[i])
            A2[i,i+1] =  (D[i  ]*(r_mid[i  ]**2)) / (r_edge[i+1] - r_edge[i  ])

        i=0
        A2[0,0] = -(D[i]*(r_mid[i]**2)) / (r_edge[i+1] - r_edge[i])
        A2[0,1] =  (D[i]*(r_mid[i]**2)) / (r_edge[i+1] - r_edge[i])

        i=N-1
        A2[i,i-1] =  (D[i-1]*(r_mid[i-1]**2)) / (r_edge[i] - r_edge[i-1])
        A2[i,i  ] = -(D[i-1]*(r_mid[i-1]**2)) / (r_edge[i] - r_edge[i-1])

        return A2


    def build_Ac_mat( self, N, D_mid, r_mid, r_edge, vols_edge ) :

        A1 = self.build_Mc_A1mat( vols_edge )
        A2 = self.build_A2_mat( N, D_mid, r_mid, r_edge, vols_edge )
        Ac = A1.dot(A2)

        return Ac


    ## Define c_e functions
    def build_Ace_mat( self, c ) :

        D_eff = self.Diff_ce( c )

        A = self.K_m.dot( flux_mat_builder( self.N, self.x_m, self.vols, D_eff ) )

        return A

    def Diff_ce( self, c ) :

        T = self.T

        D_ce = 1e-4 * 10.0**( -4.43 - (54./(T-229.-5e-3*c)) - (0.22e-3*c) )  ## Torchio (LIONSIMBA) ECS paper

        D_mid = D_ce * self.eps_eff

        if type(c) == float :
            D_edge = D_mid
        else :
            D_edge = mid_to_edge( D_mid, self.x_e )

        return D_edge

    ## Define phi_e functions
    def build_Ape_mat( self, c ) :

        k_eff = self.kapp_ce( c )

        A = flux_mat_builder( self.N, self.x_m, self.vols, k_eff )

        A[-1,-1] = 2*A[-1,-1]

        return A

    def build_Bpe_mat( self, c ) :

        gam = 2.*(1.-self.t_plus)*self.R_gas / self.F

        k_eff = self.kapp_ce( c )

        B1 = numpy.diag( 1./c ).dot( flux_mat_builder( self.N, self.x_m, self.vols, k_eff*self.T*gam ) )

        return B1

    def kapp_ce( self, c ) :

        T = self.T

        k_ce = 1e-4 * c *(   -10.5 +0.668e-3*c + 0.494e-6*c**2
                            + (0.074 - 1.78*1e-5*c - 8.86e-10*c**2)*T 
                            + (-6.96e-5 + 2.8e-8*c)*T**2 )**2  ## Torchio (LIONSIMBA) ECS paper

        k_mid = k_ce * self.eps_eff

        if type(c) == float :
            k_edge = k_mid
        else :
            k_edge = mid_to_edge( k_mid, self.x_e )

        return k_edge

    def build_Bjac_mat( self, eta, a, b ) :
            
        d = a*numpy.cosh( b*eta )*b
#        d = a*numpy.ones_like( b*eta )*b

        return numpy.diag( d )


    ## Define system equations
    def res( self, t, y, yd ) :

        ## Parse out the states
        # E-lyte conc
        ce     = y[ self.ce_inds]
        c_dots = yd[self.ce_inds]

        # Solid conc a:anode, c:cathode
        csa    = y[ self.csa_inds]
        csc    = y[ self.csc_inds]
        csa_dt = yd[self.csa_inds]
        csc_dt = yd[self.csc_inds]
        
        # E-lyte potential
        phi = y[self.pe_inds]

        # Solid potential
        phi_s_a = y[self.pa_inds]
        phi_s_c = y[self.pc_inds]

        ## Grab state dependent matrices
        # For E-lyte conc and potential (i.e., De(ce), kapp_e(ce))
        A_ce = self.build_Ace_mat( ce )
        A_pe = self.build_Ape_mat( ce )
        B_pe = self.build_Bpe_mat( ce )

        ja = self.j_a
        jc = self.j_c
        j  = self.j

        ## Compute the residuals
        # Time deriv components
        r1 = c_dots - ( ((A_ce.dot(ce)).flatten() + (self.B_ce.dot(j)).flatten()) ) # E-lyte conc

        r2 = csa_dt - (self.A_cs_a.dot(csa).flatten() + self.B_cs_a.dot(ja).flatten()) # Anode   conc
        r3 = csc_dt - (self.A_cs_c.dot(csc).flatten() + self.B_cs_c.dot(jc).flatten()) # Cathode conc
            
        r4 = A_pe.dot(phi).flatten() - B_pe.dot(ce).flatten() + self.B2_pe.dot(j).flatten() # E-lyte potential

        r5 = self.A_ps_a.dot(phi_s_a).flatten() - self.B_ps_a.dot(ja).flatten() - self.B2_ps_a*self.i_app # Anode   potential
        r6 = self.A_ps_c.dot(phi_s_c).flatten() - self.B_ps_c.dot(jc).flatten() + self.B2_ps_c*self.i_app # Cathode potential

        res_out = numpy.concatenate( [r1, r2, r3, r4, r5, r6] )

        return res_out

    def jac( self, c, t, y, yd ) :

        ### Setup 
        ## Parse out the states
        # E-lyte conc
        ce     = y[ self.ce_inds]

        ## Grab state dependent matrices
        # For E-lyte conc and potential (i.e., De(ce), kapp_e(ce))
        A_ce = self.build_Ace_mat( ce )
        A_pe = self.build_Ape_mat( ce )
        B_pe = self.build_Bpe_mat( ce )

        ###

        ### Build the Jac matrix
        ## Self coupling
        A_dots = numpy.diag( [1*c for i in range(self.num_diff_vars)] )
        j_c    = A_dots - scipy.linalg.block_diag( A_ce, self.A_cs_a, self.A_cs_c )

        j = scipy.linalg.block_diag( j_c, A_pe, self.A_ps_a, self.A_ps_c )

        j[ self.pe_inds_r, self.ce_inds_c ] = -B_pe
        ###        

        return j



csa_max = 30555.0 # [mol/m^3]
csc_max = 51554.0 # [mol/m^3]

#bsp_dir = '/home/m_klein/Projects/battsimpy/'
bsp_dir = '/home/mk-sim-linux/Battery_TempGrad/Python/batt_simulation/battsimpy/'

Ua_path = bsp_dir+'data/Model_v1/Model_Pars/solid/thermodynamics/uref_anode_bigx.csv'
Uc_path = bsp_dir+'data/Model_v1/Model_Pars/solid/thermodynamics/uref_cathode_bigx.csv'

uref_a, uref_c, duref_a, duref_c = get_smooth_Uref_data( Ua_path, Uc_path, ffa=0.4, ffc=0.2 )

xa_init, xc_init = 0.5, 0.5
ca_init = xa_init*csa_max 
cc_init = xc_init*csc_max
Ua_init = uref_a( xa_init )
Uc_init = uref_c( xc_init )

### Mesh
N = 80
Ns = int(N/8.)
Na = int(N/3.)
Nc = N - Ns - Na

X = 165e-6 # [m]

Nra = 10
Nrc = 20

Ra = 10.0e-6
Rc = 6.00e-6

cell_coated_area = 1.0 # [m^2]
I_app = 10.0 # A
#i_app = I_app / cell_coated_area # current density, [A/m^2]

### Initial conditions
# E-lyte conc
c_init = 1000.0 # [mol/m^3]
c_centered = c_init*numpy.ones( N, dtype='d' )
# E-lyte potential
p_init = 0.0 # [V]
p_centered = p_init*numpy.ones( N, dtype='d' )
# Solid potential on anode and cathode
pa_init = 0.0 # [V]
pa_centered = pa_init*numpy.ones( Na, dtype='d' )
pc_init = 0.0 # [V]
pc_centered = pc_init*numpy.ones( Nc, dtype='d' )
# Solid conc on anode and cathode
ca_centered = ca_init*numpy.ones( Na*Nra, dtype='d' )
cc_centered = cc_init*numpy.ones( Nc*Nrc, dtype='d' )

num_diff_vars = len(c_centered)+len(ca_centered)+len(cc_centered)
num_algr_vars = len(p_centered)+len(pa_centered)+len(pc_centered)

#The initial conditons
y0  = numpy.concatenate( [c_centered, ca_centered, cc_centered, p_centered, pa_centered, pc_centered] ) #Initial conditions
yd0 = [0.0 for i in range(len(y0))] #Initial conditions


#Create an Assimulo implicit problem
imp_mod = MyProblem(Na,Ns,Nc,Nra,Nrc,X,Ra,Rc,cell_coated_area,bsp_dir,y0,yd0,'Example using an analytic Jacobian')

#Sets the options to the problem
imp_mod.algvar = [1.0 for i in range(num_diff_vars)] + [0.0 for i in range(num_algr_vars)] #Set the algebraic components

#Create an Assimulo implicit solver (IDA)
imp_sim = IDA(imp_mod) #Create a IDA solver

#Sets the paramters
imp_sim.atol = 1e-5 #Default 1e-6
imp_sim.rtol = 1e-5 #Default 1e-6
imp_sim.suppress_alg = True #Suppres the algebraic variables on the error test

### Simulate
imp_mod.set_iapp( I_app/10. )
imp_sim.make_consistent('IDA_YA_YDP_INIT')
ta, ya, yda = imp_sim.simulate(0.1,5) 

imp_mod.set_iapp( I_app/2. )
imp_sim.make_consistent('IDA_YA_YDP_INIT')
tb, yb, ydb = imp_sim.simulate(0.2,5) 

imp_mod.set_iapp( I_app )
imp_sim.make_consistent('IDA_YA_YDP_INIT')
# Sim step 1
t1, y1, yd1 = imp_sim.simulate(100,100) 
#t1, y1, yd1 = imp_sim.simulate(1000,1000) 

imp_mod.set_iapp( 0.0 )
imp_sim.make_consistent('IDA_YA_YDP_INIT')
# Sim step 1
t2, y2, yd2 = imp_sim.simulate(200,100) 

print 'Performing plots...'
# extract variables
im = imp_mod
ce_1 = y1[:,im.ce_inds]
ca_1 = y1[:,im.csa_inds]
cc_1 = y1[:,im.csc_inds]
# Parse out particles
#c_s_a_list1 = [ [] for it in range(len(t1)) ]
#c_s_c_list1 = [ [] for it in range(len(t1)) ]
#for it in range(len(t1)) :
#    c_s_a_list1[it] = numpy.reshape( y1[it, imp_mod.csa_inds].T, (imp_mod.Na,imp_mod.Nra) ).T
#    c_s_c_list1[it] = numpy.reshape( y1[it, imp_mod.csc_inds].T, (imp_mod.Nc,imp_mod.Nrc) ).T


pe_1 = y1[:,im.pe_inds]
pa_1 = y1[:,im.pa_inds]
pc_1 = y1[:,im.pc_inds]

ce_2 = y2[:,im.ce_inds]
ca_2 = y2[:,im.csa_inds]
cc_2 = y2[:,im.csc_inds]
# Parse out particles
#c_s_a_list2 = [ [] for it in range(len(t2)) ]
#c_s_c_list2 = [ [] for it in range(len(t2)) ]
#for it in range(len(t2)) :
#    c_s_a_list2[it] = numpy.reshape( y2[it, imp_mod.csa_inds].T, (imp_mod.Na,imp_mod.Nra) ).T
#    c_s_c_list2[it] = numpy.reshape( y2[it, imp_mod.csc_inds].T, (imp_mod.Nc,imp_mod.Nrc) ).T

pe_2 = y2[:,im.pe_inds]
pa_2 = y2[:,im.pa_inds]
pc_2 = y2[:,im.pc_inds]

#Plot
# t1
# Plot through space
f, ax = plt.subplots(2,5)
# ce vs x
ax[0,0].plot(imp_mod.x_m*1e6,ce_1.T) 
# pe vs x
ax[0,1].plot(imp_mod.x_m*1e6,pe_1.T)
# pa vs x
ax[0,2].plot(imp_mod.x_m_a*1e6,pa_1.T)
# pc vs x
ax[0,2].plot(imp_mod.x_m_c*1e6,pc_1.T)
# ca vs x
#ax[0,3].plot(imp_mod.x_m_a*1e6,ca_1.T)
## cc vs x
#ax[0,3].plot(imp_mod.x_m_c*1e6,cc_1.T)
#for it in range(len(t1)) :
#    #cs_a
#    ax[0,3].plot( imp_mod.r_e_a, c_s_a_list1[it] )
#    #cs_c
#    ax[0,3].plot( imp_mod.r_e_a[-1]*1.2 + imp_mod.r_e_c, c_s_c_list1[it] )

ax[0,0].set_title('t1 c')
ax[0,0].set_xlabel('Cell Thickness [$\mu$m]')
ax[0,0].set_ylabel('E-lyte Conc. [mol/m$^3$]')
ax[0,1].set_title('t1 p')
ax[0,1].set_xlabel('Cell Thickness [$\mu$m]')
ax[0,1].set_ylabel('E-lyte Potential [V]')
ax[0,2].set_title('t1 p solid')
ax[0,2].set_xlabel('Cell Thickness [$\mu$m]')
ax[0,2].set_ylabel('Solid Potential [V]')
ax[0,3].set_title('t1 conc solid')
ax[0,3].set_xlabel('Cell Thickness [$\mu$m]')
ax[0,3].set_ylabel('Solid Conc. [mol/m$^3$]')

# t2
ax[1,0].plot(imp_mod.x_m*1e6,ce_2.T)
ax[1,1].plot(imp_mod.x_m*1e6,pe_2.T)

ax[1,2].plot(imp_mod.x_m_a*1e6,pa_2.T)
ax[1,2].plot(imp_mod.x_m_c*1e6,pc_2.T)

#ax[1,3].plot(imp_mod.x_m_a*1e6,ca_2.T)
##ax[1,3].plot(imp_mod.x_m_c*1e6,cc_2.T)
#for it in range(len(t2)) :
#    #cs_a
#    ax[1,3].plot( imp_mod.r_e_a, c_s_a_list2[it] )
#    #cs_c
#    ax[1,3].plot( imp_mod.r_e_a[-1]*1.2 + imp_mod.r_e_c, c_s_c_list2[it] )

ax[1,0].set_title('t2 c')
ax[1,0].set_xlabel('Cell Thickness [$\mu$m]')
ax[1,0].set_ylabel('E-lyte Conc. [mol/m$^3$]')
ax[1,1].set_title('t2 p e-lyte')
ax[1,1].set_xlabel('Cell Thickness [$\mu$m]')
ax[1,1].set_ylabel('E-lyte Potential [V]')
ax[1,2].set_title('t2 p solid')
ax[1,2].set_xlabel('Cell Thickness [$\mu$m]')
ax[1,2].set_ylabel('Solid Potential [V]')
ax[1,3].set_title('t2 Solid Conc.')
ax[1,3].set_xlabel('Cell Thickness [$\mu$m]')
ax[1,3].set_ylabel('Solid Conc. [mol/m$^3$]')

plt.tight_layout()

# Plot through time
f, ax = plt.subplots(1,4)
ax[0].plot(t1,ce_1)
ax[1].plot(t1,pe_1)
ax[2].plot(t1,pa_1) 
ax[2].plot(t1,pc_1)
ax[3].plot(t1,ca_1) 
ax[3].plot(t1,cc_1) 

ax[0].plot(t2,ce_2)
ax[1].plot(t2,pe_2)
ax[2].plot(t2,pa_2) 
ax[2].plot(t2,pc_2) 
ax[3].plot(t2,ca_2) 
ax[3].plot(t2,cc_2) 

ax[0].set_ylabel('E-lyte Conc. [mol/m$^3$]')
ax[0].set_xlabel('Time [s]')
ax[1].set_ylabel('E-lyte Potential [V]')
ax[1].set_xlabel('Time [s]')
ax[2].set_ylabel('Solid Potential [V]')
ax[2].set_xlabel('Time [s]')
ax[3].set_ylabel('Solid Conc. [mol/m$^3$]')
ax[3].set_xlabel('Time [s]')

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

#x_out[:,0] = numpy.concatenate( [c_centered, ca_centered, cc_centered] )
#z_out[:,0] = numpy.concatenate( [p_centered, pa_centered, pc_centered] )

#for it, t in enumerate(time[1:]) :

##    if it == 0 :
##        Cur_vec = [ 0.0, 0.0, 0.01*I_app ]
##    elif it == 1 :
##        Cur_vec = [ 0.0, 0.01*I_app, 0.1*I_app ]
##    elif it == 2 :
##        Cur_vec = [ 0.01*I_app, 0.1*I_app, 0.5*I_app ]
##    elif it == 3 :
##        Cur_vec = [ 0.1*I_app, 0.5*I_app, I_app ]
##    elif it == 4 :
##        Cur_vec = [ 0.5*I_app, I_app, I_app ]
##    else :
##        Cur_vec = [ I_app, I_app, I_app ]
#    if it == 0 :
#        Cur_vec = [ 0.0, 0.0, I_app ]
#    elif it == 1 :
#        Cur_vec = [ 0.0, I_app, I_app ]
#    else :
#        Cur_vec = [ I_app, I_app, I_app ]
#        
#    x_out[:,it+1], z_out[:,it+1], newtonStats = imp_mod.cn_solver( x_out[:,it], z_out[:,it], Cur_vec, delta_t )


## Parse out particles
#c_s_a_list = [ [] for it in range(len(time)) ]
#c_s_c_list = [ [] for it in range(len(time)) ]
#for it in range(len(time)) :
#    c_s_a_list[it] = numpy.reshape( x_out[imp_mod.csa_inds, it], (imp_mod.Na,imp_mod.Nra) ).T
#    c_s_c_list[it] = numpy.reshape( x_out[imp_mod.csc_inds, it], (imp_mod.Nc,imp_mod.Nrc) ).T


#plt.close()
#f, ax = plt.subplots(1,3)
## c_e
#ax[0].plot( imp_mod.x_m, x_out[:imp_mod.N,:-1] )

## phi_e
#ax[1].plot( imp_mod.x_m, z_out[:imp_mod.N,:-1] )

## phi_s
#ax[2].plot( imp_mod.x_m_a, z_out[-imp_mod.Na-imp_mod.Nc:-imp_mod.Nc,:-1] )
#ax[2].plot( imp_mod.x_m_c, z_out[-imp_mod.Nc:,:-1] )


#f2, ax2 = plt.subplots(1,2)
## cs_a
#for it in range(len(time)) :
#    #cs_a
#    ax2[0].plot( imp_mod.r_e_a, c_s_a_list[it] )
#    #cs_c
#    ax2[1].plot( imp_mod.r_e_c, c_s_c_list[it] )


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

#        ja = self.j_a
#        jc = self.j_c

#        j = self.j

#        ## Compute the residuals
#        # Time deriv components
#        r1 = ( ((A_ce.dot(ce)).flatten() + (self.B_ce.dot(j)).flatten()) ) # E-lyte conc

#        r2 = ( (self.A_cs_a.dot(csa)).flatten() + (self.B_cs_a.dot(ja)).flatten() ) # Anode   conc
#        r3 = ( (self.A_cs_c.dot(csc)).flatten() + (self.B_cs_c.dot(jc)).flatten() ) # Cathode conc

#        r4 = A_pe.dot(phi).flatten() - B_pe.dot(ce).flatten() + self.B2_pe.dot(j).flatten() # E-lyte potential

#        r5 = self.A_ps_a.dot(phi_s_a).flatten() - self.B_ps_a.dot(ja).flatten() - self.B2_ps_a*self.i_app # Anode   potential
#        r6 = self.A_ps_c.dot(phi_s_c).flatten() - self.B_ps_c.dot(jc).flatten() + self.B2_ps_c*self.i_app # Cathode potential

#        if get_mats :
#            res_out = numpy.concatenate( [r1,r2,r3] ), numpy.concatenate( [r4, r5, r6] ), { 'A_ce':A_ce, 'A_pe':A_pe, 'B_pe':B_pe }
#        else :
#            res_out = numpy.concatenate( [r1,r2,r3] ), numpy.concatenate( [r4, r5, r6] )

#        return res_out


#    def jac_system( self, mats ) :

#        A_ce = mats['A_ce'] #self.build_Ace_mat( ce )
#        A_pe = mats['A_pe'] #self.build_Ape_mat( ce )
#        B_pe = mats['B_pe'] #self.build_Bpe_mat( ce )

#        ##
#        fx = scipy.linalg.block_diag( A_ce, self.A_cs_a, self.A_cs_c )
#        ##

#        ##
#        fz = numpy.zeros( [self.num_diff_vars, self.num_algr_vars] )

#        ##
#        gx = numpy.zeros( [self.num_algr_vars, self.num_diff_vars] )
#        # phi_e vs ce
#        gx[ self.pe_inds_r2, self.ce_inds_c ] = -B_pe
#        ##

#        ##
#        # z vs z
#        gz = scipy.linalg.block_diag( A_pe, self.A_ps_a, self.A_ps_c )

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

#            F1_x = -sparse.eye(len(x)) + delta_t/2. * fx
#            F1_z = delta_t/2. * fz
#            F2_x = gx
#            F2_z = gz

#            J = numpy.concatenate( (numpy.concatenate( (F1_x, F1_z), axis=1 ), 
#                                    numpy.concatenate( (F2_x, F2_z), axis=1 )) )

#            Jsp = sparseMat( J )

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

#        return x_nxtf, z_nxtf, newtonStats
