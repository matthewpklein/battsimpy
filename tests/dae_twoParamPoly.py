import numpy
import numpy.linalg
import scipy.linalg
import scipy.interpolate

from matplotlib import pyplot as plt

import scipy.sparse as sps

from assimulo.solvers import IDA
from assimulo.problem import Implicit_Problem

from scipy.sparse.linalg import spsolve as sparseSolve
from scipy.sparse import csr_matrix as sparseMat
import scipy.sparse as sparse
import math
from copy import deepcopy


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

    def __init__(self, Na, Ns, Nc, X, Ac, bsp_dir, y0, yd0, name ) :

        Implicit_Problem.__init__(self,y0=y0,yd0=yd0,name=name)

        self.T  = 298.15 # Cell temperature, [K]
        self.Ac = Ac # Cell coated area, [m^2]

        # Control volumes and node points (mid node points and edge node points)
        self.Ns = Ns
        self.Na = Na
        self.Nc = Nc

        self.N = Na + Ns + Nc
        self.X = X

        self.num_diff_vars = N + Na + Nc
        self.num_algr_vars = Na + Nc + N + Na + Nc

        self.x_e  = numpy.linspace( 0.0, X, N+1 )
        self.x_m  = numpy.array( [ 0.5*(self.x_e[i+1]+self.x_e[i]) for i in range(N) ], dtype='d'  )
        self.vols = numpy.array( [ (self.x_e[i+1] - self.x_e[i]) for i in range(N)], dtype='d' )

        # Useful sub-meshes for the phi_s functions
        self.x_m_a = self.x_m[:Na]
        self.x_m_c = self.x_m[-Nc:]
        self.x_e_a = self.x_e[:Na+1]
        self.x_e_c = self.x_e[-Nc-1:]

        self.vols_a = self.vols[:Na]
        self.vols_c = self.vols[-Nc:]

        # Volume fraction vectors and matrices for effective parameters
        self.La, self.Ls, self.Lc = self.Na*X/self.N, self.Ns*X/self.N, self.Nc*X/self.N
        self.Na, self.Ns, self.Nc = Na, Ns, Nc
        eps_a = 0.25
        eps_s = 0.5
        eps_c = 0.2
        ba, bs, bc = 1.2, 0.5, 0.5

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

        t_plus = 0.43
        F = 96485.0

        self.t_plus = t_plus
        self.F = F
        self.R_gas = 8.314

        Rp_a = 12.0e-6
        Rp_c = 6.5e-6
        self.Rp_a = Rp_a
        self.Rp_c = Rp_c

        as_a = 3.*(1.0-numpy.array(eps_a_vec, dtype='d'))/Rp_a
        as_c = 3.*(1.0-numpy.array(eps_c_vec, dtype='d'))/Rp_c
        self.as_a = as_a
        self.as_c = as_c

        self.as_a_mean = 1./self.La*sum( [ asa*v for asa,v in zip(as_a, self.vols[:Na]) ] )
        self.as_c_mean = 1./self.Lc*sum( [ asc*v for asc,v in zip(as_c, self.vols[-Nc:]) ] )

        print 'asa diff', self.as_a_mean - as_a[0]
        print 'asc diff', self.as_c_mean - as_c[0]

        Ba = [ (1.-t_plus)*asa/ea for ea, asa in zip(eps_a_vec,as_a) ]
        Bs = [  0.0                for i in range(Ns) ]
        Bc = [ (1.-t_plus)*asc/ec for ec, asc in zip(eps_c_vec,as_c) ]

        self.B_ce = numpy.diag( numpy.array(Ba+Bs+Bc, dtype='d') )

        Bap = [ asa*F for asa in as_a  ]
        Bsp = [   0.0 for i   in range(Ns) ]
        Bcp = [ asc*F for asc in as_c  ]

        self.B2_pe = numpy.diag( numpy.array(Bap+Bsp+Bcp, dtype='d') )

        # Interpolators for De, ke
        self.De_intp  = build_interp_2d( bsp_dir+'data/Model_v1/Model_Pars/electrolyte/De.csv' )
        self.ke_intp  = build_interp_2d( bsp_dir+'data/Model_v1/Model_Pars/electrolyte/kappa.csv' )
        self.fca_intp = build_interp_2d( bsp_dir+'data/Model_v1/Model_Pars/electrolyte/fca.csv' )

        self.ce_nom = 1000.0
        
        ######
        ## Solid phase parameters and j vector matrices
        self.sig_a = 100. # [S/m]
        self.sig_c = 40. # [S/m]

        self.sig_a_eff = self.sig_a * (1.0-self.eps_a_eff)
        self.sig_c_eff = self.sig_c * (1.0-self.eps_c_eff)

        self.A_ps_a = flux_mat_builder( self.Na, self.x_m_a, numpy.ones_like(self.vols_a), self.sig_a_eff )
        self.A_ps_c = flux_mat_builder( self.Nc, self.x_m_c, numpy.ones_like(self.vols_c), self.sig_c_eff )

        # Grounding form for BCs (was only needed during testing, before BVK was incorporated for coupling
        Baps = numpy.array( [ asa*F*dxa for asa,dxa in zip(as_a, self.vols_a) ], dtype='d' )
        Bcps = numpy.array( [ asc*F*dxc for asc,dxc in zip(as_c, self.vols_c) ], dtype='d' )
        self.B_ps_a = numpy.diag( Baps )
        self.B_ps_c = numpy.diag( Bcps )

        self.B2_ps_a = numpy.zeros( self.Na, dtype='d' )
        self.B2_ps_a[ 0] = -1.
        self.B2_ps_c = numpy.zeros( self.Nc, dtype='d' )
        self.B2_ps_c[-1] = -1.

        # Two parameter Solid phase diffusion model
        Dsa = 1e-12
        Dsc = 1e-14
        self.Dsa = Dsa
        self.Dsc = Dsc

        self.csa_max = 30555.0 # [mol/m^3]
        self.csc_max = 51554.0 # [mol/m^3]

        self.B_cs_a = numpy.diag( numpy.array( [-3.0/Rp_a for i in range(Na)], dtype='d' ) ) 
        self.B_cs_c = numpy.diag( numpy.array( [-3.0/Rp_c for i in range(Nc)], dtype='d' ) ) 

        self.C_cs_a = numpy.eye(Na)
        self.C_cs_c = numpy.eye(Nc)

        self.D_cs_a = numpy.diag( numpy.array( [-Rp_a/Dsa/5.0 for i in range(Na)], dtype='d' ) ) 
        self.D_cs_c = numpy.diag( numpy.array( [-Rp_c/Dsc/5.0 for i in range(Nc)], dtype='d' ) ) 

#        bsp_dir = '/home/m_klein/Projects/battsimpy/'
#        bsp_dir = '/home/mk-sim-linux/Battery_TempGrad/Python/batt_simulation/battsimpy/'

        uref_a_map = numpy.loadtxt( bsp_dir + '/data/Model_v1/Model_Pars/solid/thermodynamics/uref_anode_x.csv'  , delimiter=',' )
        uref_c_map = numpy.loadtxt( bsp_dir + '/data/Model_v1/Model_Pars/solid/thermodynamics/uref_cathode_x.csv', delimiter=',' )

        duref_a = numpy.gradient( uref_a_map[:,1] ) / numpy.gradient( uref_a_map[:,0] )
        duref_c = numpy.gradient( uref_c_map[:,1] ) / numpy.gradient( uref_c_map[:,0] )

        if uref_a_map[1,0] > uref_a_map[0,0] :
            self.uref_a_interp  = scipy.interpolate.interp1d( uref_a_map[:,0],  uref_a_map[:,1] )
            self.duref_a_interp = scipy.interpolate.interp1d( uref_a_map[:,0], duref_a      )
        else :
            self.uref_a_interp  = scipy.interpolate.interp1d( numpy.flipud(uref_a_map[:,0]), numpy.flipud(uref_a_map[:,1]) )
            self.duref_a_interp = scipy.interpolate.interp1d( numpy.flipud(uref_a_map[:,0]), numpy.flipud(duref_a)     )

        if uref_c_map[1,0] > uref_c_map[0,0] :
            self.uref_c_interp  = scipy.interpolate.interp1d( uref_c_map[:,0], uref_c_map[:,1] )
            self.duref_c_interp = scipy.interpolate.interp1d( uref_c_map[:,0], duref_c     )
        else :
            self.uref_c_interp  = scipy.interpolate.interp1d( numpy.flipud(uref_c_map[:,0]), numpy.flipud(uref_c_map[:,1]) )
            self.duref_c_interp = scipy.interpolate.interp1d( numpy.flipud(uref_c_map[:,0]), numpy.flipud(duref_c)     )

        # Plot the Uref data for verification
#        xa = numpy.linspace( 0.05, 0.95, 50 )
#        xc = numpy.linspace( 0.40, 0.95, 50 )
#        plt.figure()
#        plt.plot( uref_a_map[:,0], uref_a_map[:,1], label='Ua map' )
#        plt.plot( uref_c_map[:,0], uref_c_map[:,1], label='Uc map' )

#        plt.plot( xa, self.uref_a_interp(xa), label='Ua interp' )
#        plt.plot( xc, self.uref_c_interp(xc), label='Uc interp' )
#        plt.legend()

#        plt.figure()
#        plt.plot( uref_a_map[:,0], duref_a, label='dUa map' )
#        plt.plot( uref_c_map[:,0], duref_c, label='dUc map' )

#        plt.plot( xa, self.duref_a_interp(xa), label='dUa interp' )
#        plt.plot( xc, self.duref_c_interp(xc), label='dUc interp' )
#        plt.legend()

#        plt.show()

        # Reaction kinetics parameters
        self.io_a = 5.0 # [A/m^2]
        self.io_c = 5.0 # [A/m^2]

        # System indices
        self.ce_inds  = range(self.N)
        self.csa_inds = range(self.N, self.N+self.Na)
        self.csc_inds = range(self.N+self.Na, self.N+self.Na+self.Nc)

        c_end = self.N+self.Na+self.Nc

        self.ja_inds = range(c_end, c_end+self.Na)
        self.jc_inds = range(c_end+self.Na, c_end+self.Na +self.Nc)
        
        self.pe_inds   = range( c_end+self.Na+self.Nc, c_end+self.Na+self.Nc +self.N )
        self.pe_a_inds = range( c_end+self.Na+self.Nc, c_end+self.Na+self.Nc +self.Na )
        self.pe_c_inds = range( c_end+self.Na+self.Nc +self.Na+self.Ns, c_end+self.Na+self.Nc +self.N )

        self.pa_inds = range( c_end+self.Na+self.Nc+self.N, c_end+self.Na+self.Nc+self.N +self.Na )
        self.pc_inds = range( c_end+self.Na+self.Nc+self.N+self.Na, c_end+self.Na+self.Nc+self.N+self.Na +self.Nc )

    def set_iapp( self, I_app ) :
        self.i_app = I_app / self.Ac

    ## Define c_e functions
    def build_Ace_mat( self, c ) :

        D_eff = self.Diff_ce( c )

        A = self.K_m.dot( flux_mat_builder( self.N, self.x_m, self.vols, D_eff ) )

        return A

    def Diff_ce( self, c ) :

        T = self.T

#        D_ce = 1e-4 * 10.0**( -4.43 - (54./(T-229.-5e-3*c)) - (0.22e-3*c) )  ## Torchio (LIONSIMBA) ECS paper

        D_ce = self.De_intp( c, T, grid=False ).flatten()
        
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

        gam = 2.*(1.-self.t_plus)*self.R_gas*self.T / self.F

        k_eff = self.kapp_ce( c )

        c_edge = mid_to_edge( c, self.x_e )

        B1 = flux_mat_builder( self.N, self.x_m, self.vols, k_eff*gam/c_edge )

        return B1

    def kapp_ce( self, c ) :

        T = self.T

#        k_ce = 1e-4 * c *(   -10.5 +0.668e-3*c + 0.494e-6*c**2
#                            + (0.074 - 1.78*1e-5*c - 8.86e-10*c**2)*T 
#                            + (-6.96e-5 + 2.8e-8*c)*T**2 )**2  ## Torchio (LIONSIMBA) ECS paper

        k_ce = 1e-1*self.ke_intp( c, T, grid=False ).flatten() # 1e-1 converts from mS/cm to S/m (model uses SI units)

        k_mid = k_ce * self.eps_eff

        if type(c) == float :
            k_edge = k_mid
        else :
            k_edge = mid_to_edge( k_mid, self.x_e )

        return k_edge

    def build_Bjac_mat( self, eta, a, b ) :
            
        d = a*numpy.cosh( b*eta )*b

        return numpy.diag( d )


    def get_voltage( self, y ) :
        """
        Return the cell potential
        """
        pc = y[self.pc_inds]
        pa = y[self.pa_inds]

        Vcell = pc[-1] - pa[0]

        return Vcell


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

        # Reaction (Butler-Volmer Kinetics)
        ja_rxn = y[self.ja_inds]
        jc_rxn = y[self.jc_inds]

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

        ## Compute extra variables
        # For the reaction kinetics
        csa_ss = csa + (self.D_cs_a.dot( ja_rxn ).flatten()) # anode   surface conc
        csc_ss = csc + (self.D_cs_c.dot( jc_rxn ).flatten()) # cathode surface conc

        Uref_a = self.uref_a_interp( csa_ss/self.csa_max ) # anode   equilibrium potential
        Uref_c = self.uref_c_interp( csc_ss/self.csc_max ) # cathode equilibrium potential

        eta_a  = phi_s_a - phi[:self.Na]  - Uref_a  # anode   overpotential
        eta_c  = phi_s_c - phi[-self.Nc:] - Uref_c  # cathode overpotential

#        ja = 2.0*self.io_a * numpy.sqrt( ce[:self.Na]/self.ce_nom * (1.0 - csa_ss/self.csa_max) * (csa_ss/self.csa_max) ) * numpy.sinh( self.R_gas/(2.0*self.F*self.T)*eta_a )
#        jc = 2.0*self.io_c * numpy.sqrt( ce[-self.Nc:]/self.ce_nom * (1.0 - csc_ss/self.csc_max) * (csc_ss/self.csc_max) ) * numpy.sinh( self.R_gas/(2.0*self.F*self.T)*eta_c )
        ja = (2.0*self.io_a/self.F) * numpy.sinh( 0.5*self.F/(self.R_gas*self.T)*eta_a )
        jc = (2.0*self.io_c/self.F) * numpy.sinh( 0.5*self.F/(self.R_gas*self.T)*eta_c )

        j = numpy.concatenate( [ ja_rxn, numpy.zeros(self.Ns), jc_rxn ] )

        ## Compute the residuals
        # Time deriv components
        r1 = c_dots - ( ((A_ce.dot(ce)).flatten() + (self.B_ce.dot(j)).flatten()) ) # E-lyte conc

        r2 = csa_dt - (self.B_cs_a.dot(ja_rxn).flatten()) # Anode   conc
        r3 = csc_dt - (self.B_cs_c.dot(jc_rxn).flatten()) # Cathode conc
            
        # Algebraic components
        r4 = ja_rxn - ja
        r5 = jc_rxn - jc 

        r6 = A_pe.dot(phi).flatten() - B_pe.dot(ce).flatten() + self.B2_pe.dot(j).flatten() # E-lyte potential

        r7 = self.A_ps_a.dot(phi_s_a).flatten() - self.B_ps_a.dot(ja_rxn).flatten() - self.B2_ps_a*self.i_app # Anode   potential #+ extra #
        r8 = self.A_ps_c.dot(phi_s_c).flatten() - self.B_ps_c.dot(jc_rxn).flatten() + self.B2_ps_c*self.i_app # Cathode potential

        res_out = numpy.concatenate( [r1, r2, r3, r4, r5, r6, r7, r8] )

        return res_out

    def jac( self, c, t, y, yd ) :

        ### Setup 
        ## Parse out the states
        # E-lyte conc
        ce     = y[ self.ce_inds]
#        c_dots = yd[self.ce_inds]

        # Solid conc a:anode, c:cathode
        csa    = y[ self.csa_inds]
        csc    = y[ self.csc_inds]
#        csa_dt = yd[self.csa_inds]
#        csc_dt = yd[self.csc_inds]

        # Reaction (Butler-Volmer Kinetics)
        ja_rxn = y[self.ja_inds]
        jc_rxn = y[self.jc_inds]

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

        ## Compute extra variables
        # For the reaction kinetics
        csa_ss = csa + (self.D_cs_a.dot( ja_rxn ).flatten()) # anode   surface conc
        csc_ss = csc + (self.D_cs_c.dot( jc_rxn ).flatten()) # cathode surface conc

        Uref_a = self.uref_a_interp( csa_ss/self.csa_max ) # anode   equilibrium potential
        Uref_c = self.uref_c_interp( csc_ss/self.csc_max ) # cathode equilibrium potential

        eta_a  = phi_s_a - phi[:self.Na]  - Uref_a  # anode   overpotential
        eta_c  = phi_s_c - phi[-self.Nc:] - Uref_c  # cathode overpotential
        ###

        ### Build the Jac matrix
        ## Self coupling
        A_dots = numpy.diag( [1*c for i in range(self.N+self.Na+self.Nc)] )
        j_c    = A_dots - scipy.linalg.block_diag( A_ce, numpy.zeros([self.Na,self.Na]), numpy.zeros([self.Nc,self.Nc]) )

        Bjac_a = self.build_Bjac_mat( eta_a, (2.0*self.io_a/self.F), 0.5*self.F/(self.R_gas*self.T) )
        Bjac_c = self.build_Bjac_mat( eta_c, (2.0*self.io_c/self.F), 0.5*self.F/(self.R_gas*self.T) )

        DUDcsa_ss = numpy.diag( (1.0/self.csa_max)*self.duref_a_interp(csa_ss/self.csa_max) )
        DUDcsc_ss = numpy.diag( (1.0/self.csc_max)*self.duref_c_interp(csc_ss/self.csc_max) )

        A_ja = numpy.diag(numpy.ones(self.Na)) - Bjac_a.dot(DUDcsa_ss.dot(-1.0*self.D_cs_a))
        A_jc = numpy.diag(numpy.ones(self.Nc)) - Bjac_c.dot(DUDcsc_ss.dot(-1.0*self.D_cs_c))

        j = scipy.linalg.block_diag( j_c, A_ja, A_jc, A_pe, self.A_ps_a, self.A_ps_c )

        ## Cross coupling
        # c_e: j coupling back in
        j[ numpy.ix_(self.ce_inds, self.ja_inds) ] = -self.B_ce[:, :self.Na]
        j[ numpy.ix_(self.ce_inds, self.jc_inds) ] = -self.B_ce[:, -self.Nc:]

        # cs_a: j coupling
        j[ numpy.ix_(self.csa_inds, self.ja_inds) ] = -self.B_cs_a
        # cs_c: j coupling
        j[ numpy.ix_(self.csc_inds, self.jc_inds) ] = -self.B_cs_c

        # j_a: pe, pa, csa  coupling
        j[numpy.ix_(self.ja_inds, self.pa_inds  )] = -Bjac_a*( 1.0)
        j[numpy.ix_(self.ja_inds, self.pe_a_inds)] = -Bjac_a*(-1.0)
        j[numpy.ix_(self.ja_inds, self.csa_inds )] = -Bjac_a.dot(-1.0*DUDcsa_ss*1.0)
#        j[numpy.ix_(self.ja_inds, self.ja_inds ) ] = j[numpy.ix_(self.ja_inds, self.ja_inds ) ] - Bjac_a.dot(DUDcsa_ss.dot(self.D_cs_a)*(-1.0)))
        # j_c: pe, pc, csc  coupling         
        j[numpy.ix_(self.jc_inds, self.pc_inds  )] = -Bjac_c*( 1.0)
        j[numpy.ix_(self.jc_inds, self.pe_c_inds)] = -Bjac_c*(-1.0)
        j[numpy.ix_(self.jc_inds, self.csc_inds )] = -Bjac_c.dot(-1.0*DUDcsc_ss*1.0)
#        j[numpy.ix_(self.jc_inds, self.jc_inds ) ] = j[numpy.ix_(self.jc_inds, self.jc_inds ) ] - Bjac_c.dot(DUDcsc_ss.dot(self.D_cs_c)*(-1.0)))

        # phi_e: ce coupling into phi_e equation
        j[numpy.ix_(self.pe_inds,self.ce_inds)] = -B_pe
        j[numpy.ix_(self.pe_inds,self.ja_inds)] = self.B2_pe[:,:self.Na]
        j[numpy.ix_(self.pe_inds,self.jc_inds)] = self.B2_pe[:,-self.Nc:]

        # phi_s_a: ja
        j[numpy.ix_(self.pa_inds,self.ja_inds)] = -self.B_ps_a
        # phi_s_c: jc
        j[numpy.ix_(self.pc_inds,self.jc_inds)] = -self.B_ps_c
        ###        

        return j


csa_max = 30555.0 # [mol/m^3]
csc_max = 51554.0 # [mol/m^3]

bsp_dir = '/home/m_klein/Projects/battsimpy/'
#bsp_dir = '/home/mk-sim-linux/Battery_TempGrad/Python/batt_simulation/battsimpy/'
#bsp_dir = '/Users/mk/Desktop/battsim/battsimpy/'

uref_a_map = numpy.loadtxt( bsp_dir + '/data/Model_v1/Model_Pars/solid/thermodynamics/uref_anode_x.csv'  , delimiter=',' )
uref_c_map = numpy.loadtxt( bsp_dir + '/data/Model_v1/Model_Pars/solid/thermodynamics/uref_cathode_x.csv', delimiter=',' )

if uref_a_map[1,0] > uref_a_map[0,0] :
    uref_a_interp = scipy.interpolate.interp1d( uref_a_map[:,0], uref_a_map[:,1] )
else :
    uref_a_interp = scipy.interpolate.interp1d( numpy.flipud(uref_a_map[:,0]), numpy.flipud(uref_a_map[:,1]) )

if uref_c_map[1,0] > uref_c_map[0,0] :
    uref_c_interp = scipy.interpolate.interp1d( uref_c_map[:,0], uref_c_map[:,1] )
else :
    uref_c_interp = scipy.interpolate.interp1d( numpy.flipud(uref_c_map[:,0]), numpy.flipud(uref_c_map[:,1]) )

xa_init, xc_init = 0.8, 0.37
ca_init = xa_init*csa_max 
cc_init = xc_init*csc_max
Ua_init = uref_a_interp( xa_init )
Uc_init = uref_c_interp( xc_init )

print Ua_init
print Uc_init

### Mesh
La = 65.0
Ls = 25.0
Lc = 55.0
Lt = (La+Ls+Lc)
X = Lt*1e-6 # [m]

N = 80
Ns = int(N*(Ls/Lt))
Na = int(N*(La/Lt))
Nc = N - Ns - Na


Crate = 1.
Vcut  = 3.0 # [V], cutoff voltage for end of discharge
ce_lims = [100.,3000.]

cell_cap = 29.0
cell_coated_area = 1.0 # [m^2]

I_app = Crate*cell_cap # A

### Initial conditions
# E-lyte conc
c_init = 1100.0 # [mol/m^3]
c_centered = c_init*numpy.ones( N, dtype='d' ) #numpy.linspace(1500, 500, N) #
# E-lyte potential
p_init = 0.0 # [V]
p_centered = p_init*numpy.ones( N, dtype='d' )
# Solid potential on anode and cathode
pa_init = Ua_init #0.0 # [V]
pa_centered = pa_init*numpy.ones( Na, dtype='d' )
pc_init = Uc_init#-Ua_init #0.0 # [V]
pc_centered = pc_init*numpy.ones( Nc, dtype='d' )
# Solid conc on anode and cathode
#ca_init = 10000.0 # [mol/m^3]
ca_centered = ca_init*numpy.ones( Na, dtype='d' )
#cc_init = 30000.0 # [mol/m^3]
cc_centered = cc_init*numpy.ones( Nc, dtype='d' )

ja = numpy.zeros(Na)
jc = numpy.zeros(Nc)

#The initial conditons
y0  = numpy.concatenate( [c_centered, ca_centered, cc_centered, ja, jc, p_centered, pa_centered, pc_centered] ) #Initial conditions
yd0 = [0.0 for i in range(N+Na+Nc +Na+Nc +N+Na+Nc)] #Initial conditions

#Create an Assimulo implicit problem
imp_mod = MyProblem(Na,Ns,Nc,X,cell_coated_area,bsp_dir,y0,yd0,'Example using an analytic Jacobian')

#Sets the options to the problem
imp_mod.algvar = [1.0 for i in range(N+Na+Nc)] + [0.0 for i in range(Na+Nc +N+Na+Nc)] #Set the algebraic components

#Create an Assimulo implicit solver (IDA)
imp_sim = IDA(imp_mod) #Create a IDA solver

#Sets the paramters
imp_sim.atol = 1e-5 #Default 1e-6
imp_sim.rtol = 1e-5 #Default 1e-6
imp_sim.suppress_alg = True #Suppres the algebraic variables on the error test

### Simulate
#imp_mod.set_iapp( I_app/10. )
#imp_sim.make_consistent('IDA_YA_YDP_INIT')
#ta, ya, yda = imp_sim.simulate(0.1,5) 
##
#imp_mod.set_iapp( I_app/2. )
#imp_sim.make_consistent('IDA_YA_YDP_INIT')
#tb, yb, ydb = imp_sim.simulate(0.2,5) 

#imp_mod.set_iapp( I_app )
#imp_sim.make_consistent('IDA_YA_YDP_INIT')
## Sim step 1
#t1, y1, yd1 = imp_sim.simulate(1./Crate*3600.*0.2,100) 

imp_sim.display_progress = False
imp_sim.verbosity = 50
imp_sim.report_continuously = True
imp_sim.time_limit = 10.

### Simulate
t01, t02 = 0.1, 0.2

imp_mod.set_iapp( I_app/10. )
imp_sim.make_consistent('IDA_YA_YDP_INIT')
ta, ya, yda = imp_sim.simulate(t01,2) 

imp_mod.set_iapp( I_app/2. )
imp_sim.make_consistent('IDA_YA_YDP_INIT')
tb, yb, ydb = imp_sim.simulate(t02,2) 

print 'yb shape', yb.shape

# Sim step 1
#imp_mod.set_iapp( I_app )
#imp_sim.make_consistent('IDA_YA_YDP_INIT')
#t1, y1, yd1 = imp_sim.simulate(1.0/Crate*3600.0,100) 

NT = 30
time   = numpy.linspace( t02+0.1, 1.0/Crate*3600.0*0.3, NT )#numpy.linspace( t02+0.1, 60., NT ) #
t_out  = [ 0 for ts in time ]
V_out  = [ 0 for ts in time ]
y_out  = numpy.zeros( [len(time), yb.shape[ 1]] )
yd_out = numpy.zeros( [len(time), ydb.shape[1]] )

print 'y_out.shape', y_out.shape

it = 0
V_cell = imp_mod.get_voltage( yb[-1,:].flatten() )
ce_now = yb[-1,imp_mod.ce_inds].flatten()
print 'V_cell prior to time loop:', V_cell

imp_mod.set_iapp( I_app )
imp_sim.make_consistent('IDA_YA_YDP_INIT')
sim_stopped = 0
while V_cell > Vcut and max(ce_now)<max(ce_lims) and min(ce_now)>min(ce_lims) and not sim_stopped and it<len(time) :

    try :
        ti, yi, ydi = imp_sim.simulate(time[it],1)
    except :
        ti  = [t_out[it-1],t_out[it-1]]
        yi  = y_out[ it-2:it,:]
        ydi = yd_out[ it-2:it,:]

        sim_stopped = 1

        print 'Sim stopped due time integration failure.'

    t_out[ it]   = ti[ -1  ]
    y_out[ it,:] = yi[ -1,:]
    yd_out[it,:] = ydi[-1,:]

    V_cell = imp_mod.get_voltage( y_out[it,:] )

    V_out[it] = V_cell

    ce_now = y_out[it,imp_mod.ce_inds]

    print 'time:',round(t_out[it],3), ' |  Voltage:', round(V_cell,3)

    if V_cell < Vcut :
        print '\n','Vcut stopped simulation.'
    elif max(ce_now)>max(ce_lims) :
        print '\n','ce max stopped simulation.'
    elif min(ce_now)<min(ce_lims) :
        print '\n','ce min stopped simulation.'

    it+=1

if it < len(time) :
    t_out  = t_out[ :it  ]
    V_out  = V_out[ :it  ]
    y_out  = y_out[ :it,:]
    yd_out = yd_out[:it,:]

ce = y_out[:,imp_mod.ce_inds]

f,ax=plt.subplots(1,2)
ax[0].plot( imp_mod.x_m, ce.T )
ax[1].plot( t_out, V_out )
plt.show()


t1  = t_out
y1  = y_out
yd1 = yd_out


imp_mod.set_iapp( 0.0 )
imp_sim.make_consistent('IDA_YA_YDP_INIT')
# Sim step 1
t2, y2, yd2 = imp_sim.simulate(t1[-1]*1.5,100) 


c_avg_0 = numpy.mean( imp_mod.eps_m*y0[:N] )
c_avg_f = numpy.mean( imp_mod.eps_m*y2[-1,:N] )

print c_avg_0
print c_avg_f


# extract variables
im = imp_mod
ce_1 = y1[:,im.ce_inds]
ca_1 = y1[:,im.csa_inds]
cc_1 = y1[:,im.csc_inds]

pe_1 = y1[:,im.pe_inds]
pa_1 = y1[:,im.pa_inds]
pc_1 = y1[:,im.pc_inds]

ja_1 = y1[:,im.ja_inds]
jc_1 = y1[:,im.jc_inds]

ce_2 = y2[:,im.ce_inds]
ca_2 = y2[:,im.csa_inds]
cc_2 = y2[:,im.csc_inds]

pe_2 = y2[:,im.pe_inds]
pa_2 = y2[:,im.pa_inds]
pc_2 = y2[:,im.pc_inds]

ja_2 = y2[:,im.ja_inds]
jc_2 = y2[:,im.jc_inds]

Jsum_a1 = numpy.array( [ sum(imp_mod.vols_a*imp_mod.F*imp_mod.as_a*ja_1[i,:]) for i in range(len(ja_1[:,0])) ] )
Jsum_c1 = numpy.array( [ sum(imp_mod.vols_c*imp_mod.F*imp_mod.as_c*jc_1[i,:]) for i in range(len(jc_1[:,0])) ] )

plt.figure()
plt.plot( t1, Jsum_a1-Jsum_c1 )

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
ax[0,3].plot(imp_mod.x_m_a*1e6,ca_1.T)
# cc vs x
ax[0,3].plot(imp_mod.x_m_c*1e6,cc_1.T)
# ja vs x
ax[0,4].plot(imp_mod.x_m_a*1e6,ja_1.T)
# jc vs x
ax[0,4].plot(imp_mod.x_m_c*1e6,jc_1.T)

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

ax[1,3].plot(imp_mod.x_m_a*1e6,ca_2.T)
ax[1,3].plot(imp_mod.x_m_c*1e6,cc_2.T)

ax[1,4].plot(imp_mod.x_m_a*1e6,ja_2.T)
ax[1,4].plot(imp_mod.x_m_c*1e6,jc_2.T)

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





#imp_mod = MyProblem(Na,Ns,Nc,X,cell_coated_area,bsp_dir,y0,yd0,'Example using an analytic Jacobian')

## my own time solver

#delta_t = 1.0
#tf = 100.
#time = [ i*delta_t for i in range(int(tf/delta_t)+1) ]

#print time

#x_out = numpy.zeros( [imp_mod.N+imp_mod.Na+imp_mod.Nc, len(time)] )
#z_out = numpy.zeros( [imp_mod.Na+imp_mod.Nc+imp_mod.N+imp_mod.Na+imp_mod.Nc, len(time)] )

#x_out[:,0] = numpy.concatenate( [c_centered, ca_centered, cc_centered] )
#z_out[:,0] = numpy.concatenate( [ja, jc, p_centered, pa_centered, pc_centered] )

#for it, t in enumerate(time[1:]) :

#    if it == 0 :
#        Cur_vec = [ 0.0, 0.0, 0.1*I_app ]
#    elif it == 1 :
#        Cur_vec = [ 0.0, 0.1*I_app, 0.5*I_app ]
#    elif it == 2 :
#        Cur_vec = [ 0.1*I_app, 0.5*I_app, I_app ]
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




#    def dae_system_num( self, y ) :

#        self.set_iapp( self.Input )

#        ## Parse out the states
#        # E-lyte conc
#        ce     = y[numpy.ix_( self.ce_inds)]

#        # Solid conc a:anode, c:cathode
#        csa    = y[numpy.ix_( self.csa_inds)]
#        csc    = y[numpy.ix_( self.csc_inds)]

#        # Reaction (Butler-Volmer Kinetics)
#        ja_rxn = y[numpy.ix_(self.ja_inds)]
#        jc_rxn = y[numpy.ix_(self.jc_inds)]

#        # E-lyte potential
#        phi = y[numpy.ix_(self.pe_inds)]

#        # Solid potential
#        phi_s_a = y[numpy.ix_(self.pa_inds)]
#        phi_s_c = y[numpy.ix_(self.pc_inds)]

#        ## Grab state dependent matrices
#        # For E-lyte conc and potential (i.e., De(ce), kapp_e(ce))
#        A_ce = self.build_Ace_mat( ce )
#        A_pe = self.build_Ape_mat( ce )
#        B_pe = self.build_Bpe_mat( ce )

#        ## Compute extra variables
#        # For the reaction kinetics
#        csa_ss = csa + (self.D_cs_a.dot( ja_rxn ).flatten()) # anode   surface conc
#        csc_ss = csc + (self.D_cs_c.dot( jc_rxn ).flatten()) # cathode surface conc

#        xa    = csa   /self.csa_max
#        xc    = csc   /self.csc_max
#        xa_ss = csa_ss/self.csa_max
#        xc_ss = csc_ss/self.csc_max
#        
#        Uref_a = self.uref_a_interp( xa_ss ) # anode   equilibrium potential
#        Uref_c = self.uref_c_interp( xc_ss ) # cathode equilibrium potential

#        eta_a  = phi_s_a - phi[:self.Na]  - Uref_a  # anode   overpotential
#        eta_c  = phi_s_c - phi[-self.Nc:] - Uref_c  # cathode overpotential

#        ja = 2.0*self.io_a/self.F * numpy.sinh( 0.5*self.F/(self.R_gas*self.T)*eta_a )
#        jc = 2.0*self.io_c/self.F * numpy.sinh( 0.5*self.F/(self.R_gas*self.T)*eta_c )

#        j = numpy.concatenate( [ ja_rxn, numpy.zeros(self.Ns), jc_rxn ] )

#        ## Compute the residuals
#        # Time deriv components
#        r1 = ( ((A_ce.dot(ce)).flatten() + (self.B_ce.dot(j)).flatten()) ) # E-lyte conc

#        r2 = (self.B_cs_a.dot(ja_rxn).flatten()) # Anode   conc
#        r3 = (self.B_cs_c.dot(jc_rxn).flatten()) # Cathode conc
#            
#        # Algebraic components
#        r4 = ja_rxn - ja
#        r5 = jc_rxn - jc 

#        r6 = A_pe.dot(phi).flatten() - B_pe.dot(ce).flatten() + self.B2_pe.dot(j).flatten() # E-lyte potential

#        r7 = self.A_ps_a.dot(phi_s_a).flatten() - self.B_ps_a.dot(ja_rxn).flatten() - self.B2_ps_a*self.i_app # Anode   potential
#        r8 = self.A_ps_c.dot(phi_s_c).flatten() - self.B_ps_c.dot(jc_rxn).flatten() + self.B2_ps_c*self.i_app # Cathode potential

#        res_out = numpy.concatenate( [r1,r2,r3, r4, r5, r6, r7, r8] )

#        return res_out



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

#        ## Compute extra variables
#        # For the reaction kinetics
#        csa_ss = csa + (self.D_cs_a.dot( ja_rxn ).flatten()) # anode   surface conc
#        csc_ss = csc + (self.D_cs_c.dot( jc_rxn ).flatten()) # cathode surface conc

#        xa    = csa   /self.csa_max
#        xc    = csc   /self.csc_max
#        xa_ss = csa_ss/self.csa_max
#        xc_ss = csc_ss/self.csc_max

#        Uref_a = self.uref_a_interp( xa_ss ) # anode   equilibrium potential
#        Uref_c = self.uref_c_interp( xc_ss ) # cathode equilibrium potential

#        eta_a  = phi_s_a - phi[:self.Na]  - Uref_a  # anode   overpotential
#        eta_c  = phi_s_c - phi[-self.Nc:] - Uref_c  # cathode overpotential

##        ja = 2.0*self.io_a * numpy.sqrt( ce[:self.Na]/self.ce_nom * (1.0 - csa_ss/self.csa_max) * (csa_ss/self.csa_max) ) * numpy.sinh( self.R_gas/(2.0*self.F*self.T)*eta_a )
##        jc = 2.0*self.io_c * numpy.sqrt( ce[-self.Nc:]/self.ce_nom * (1.0 - csc_ss/self.csc_max) * (csc_ss/self.csc_max) ) * numpy.sinh( self.R_gas/(2.0*self.F*self.T)*eta_c )
#        ja = 2.0*self.io_a/self.F * numpy.sinh( 0.5*self.F/(self.R_gas*self.T)*eta_a )
#        jc = 2.0*self.io_c/self.F * numpy.sinh( 0.5*self.F/(self.R_gas*self.T)*eta_c )

#        j = numpy.concatenate( [ ja_rxn, numpy.zeros(self.Ns), jc_rxn ] )

##        plt.figure()    
##        plt.plot( self.x_m, j )
##        plt.show()

#        ## Compute the residuals
#        # Time deriv components
#        r1 = ( ((A_ce.dot(ce)).flatten() + (self.B_ce.dot(j)).flatten()) ) # E-lyte conc

#        r2 = (self.B_cs_a.dot(ja_rxn).flatten()) # Anode   conc
#        r3 = (self.B_cs_c.dot(jc_rxn).flatten()) # Cathode conc
#            
#        # Algebraic components
#        r4 = ja_rxn - ja
#        r5 = jc_rxn - jc 

#        r6 = A_pe.dot(phi).flatten() - B_pe.dot(ce).flatten() + self.B2_pe.dot(j).flatten() # E-lyte potential

#        r7 = self.A_ps_a.dot(phi_s_a).flatten() - self.B_ps_a.dot(ja_rxn).flatten() - self.B2_ps_a*self.i_app # Anode   potential
#        r8 = self.A_ps_c.dot(phi_s_c).flatten() - self.B_ps_c.dot(jc_rxn).flatten() + self.B2_ps_c*self.i_app # Cathode potential

#        if get_mats :
#            res_out = numpy.concatenate( [r1,r2,r3] ), numpy.concatenate( [r4, r5, r6, r7, r8] ), { 'A_ce':A_ce, 'A_pe':A_pe, 'B_pe':B_pe, 'csa':csa, 'csc':csc, 'csa_ss':csa_ss, 'csc_ss':csc_ss, 'xa':xa, 'xc':xc, 'xa_ss':xa_ss, 'xc_ss':xc_ss, 'eta_a':eta_a, 'eta_c':eta_c }
#        else :
#            res_out = numpy.concatenate( [r1,r2,r3] ), numpy.concatenate( [r4, r5, r6, r7, r8] )

#        return res_out


#    def jac_system( self, mats ) :

#        A_ce = mats['A_ce'] #self.build_Ace_mat( ce )
#        A_pe = mats['A_pe'] #self.build_Ape_mat( ce )
#        B_pe = mats['B_pe'] #self.build_Bpe_mat( ce )

#        Bjac_a = self.build_Bjac_mat( mats['eta_a'], 2.0*self.io_a/self.F, 0.5*self.F/(self.R_gas*self.T) )
#        Bjac_c = self.build_Bjac_mat( mats['eta_c'], 2.0*self.io_c/self.F, 0.5*self.F/(self.R_gas*self.T) )

#        DUDcsa_ss = numpy.diag( (1.0/self.csa_max)*self.duref_a_interp(mats['xa_ss']) )
#        DUDcsc_ss = numpy.diag( (1.0/self.csc_max)*self.duref_c_interp(mats['xc_ss']) )

#        Bja = Bjac_a.dot(-1.0*DUDcsa_ss.dot(self.D_cs_a))
#        Bjc = Bjac_c.dot(-1.0*DUDcsc_ss.dot(self.D_cs_c))

#        A_ja = numpy.diag(numpy.ones(self.Na)) - Bja
#        A_jc = numpy.diag(numpy.ones(self.Nc)) - Bjc

#        ##
#        fx =  scipy.linalg.block_diag( A_ce, numpy.zeros([self.Na,self.Na]), numpy.zeros([self.Nc,self.Nc]) )
#        ##

#        ##
#        fz =  numpy.zeros( [self.N+self.Na+self.Nc, self.Na+self.Nc + self.N+self.Na+self.Nc] )

#        # ce vs j
#        fz[ numpy.ix_(range(self.N), range(self.Na)) ] = self.B_ce[:, :self.Na]
#        fz[ numpy.ix_(range(self.N), range(self.Na,self.Na+self.Nc)) ] = self.B_ce[:, -self.Nc:]

#        # cs vs j
#        fz[ numpy.ix_(range(self.N,self.N+self.Na), range(self.Na)) ] = self.B_cs_a
#        fz[ numpy.ix_(range(self.N+self.Na,self.N+self.Na+self.Nc), range(self.Na,self.Na+self.Nc)) ] = self.B_cs_c
#        ##

#        ##
#        gx =  numpy.zeros( [self.Na+self.Nc + self.N+self.Na+self.Nc, self.N+self.Na+self.Nc] )

#        # j vs cs_bar
#        gx[numpy.ix_(range(self.Na),range(self.N,self.N+self.Na))] = -Bjac_a.dot(-1.0*DUDcsa_ss*1.0)
#        gx[numpy.ix_(range(self.Na,self.Na+self.Nc),range(self.N+self.Na,self.N+self.Na+self.Nc))] = -Bjac_c.dot(-1.0*DUDcsc_ss*1.0)

#        # phi_e vs ce
#        gx[numpy.ix_(range(self.Na+self.Nc,self.Na+self.Nc+self.N),range(self.N))] = -B_pe
#        ##

#        ##
#        # z vs z
#        gz0 =  scipy.linalg.block_diag( A_ja, A_jc, A_pe, self.A_ps_a, self.A_ps_c )

#        # z cross coupling
#        gz00 = numpy.zeros_like( gz0 )
#        # phi_e vs j
#        gz00[ numpy.ix_(range(self.Na+self.Nc,self.Na+self.Nc+self.N),range(self.Na)) ] = self.B2_pe[:,:self.Na]
#        gz00[ numpy.ix_(range(self.Na+self.Nc,self.Na+self.Nc+self.N),range(self.Na,self.Na+self.Nc)) ] = self.B2_pe[:,-self.Nc:]

#        # phi_s vs j
#        gz00[ numpy.ix_(range(self.Na+self.Nc+self.N, self.Na+self.Nc+self.N +self.Na),range(self.Na)) ] = -self.B_ps_a
#        gz00[ numpy.ix_(range(self.Na+self.Nc+self.N+self.Na,self.Na+self.Nc+self.N+self.Na+self.Nc),range(self.Na,self.Na+self.Nc)) ] = -self.B_ps_c

#        # j vs phi_s
#        gz00[ numpy.ix_(range(self.Na), range(self.Na+self.Nc+self.N,self.Na+self.Nc+self.N+self.Na)) ] = -Bjac_a*( 1.0)
#        gz00[ numpy.ix_(range(self.Na,self.Na+self.Nc),range(self.Na+self.Nc+self.N+self.Na,self.Na+self.Nc+self.N+self.Na+self.Nc)) ] = -Bjac_c*( 1.0)

#        # j vs phi_e
#        gz00[ numpy.ix_(range(self.Na), range(self.Na+self.Nc,self.Na+self.Nc+self.Na)) ] = -Bjac_a*(-1.0)
#        gz00[ numpy.ix_(range(self.Na,self.Na+self.Nc), range(self.Na+self.Nc+self.Na+self.Ns,self.Na+self.Nc+self.N)) ] = -Bjac_c*(-1.0)

#        gz = gz0 + gz00

#        return fx, fz, gx, gz


#    def cn_solver( self, x, z, Cur_vec, delta_t ) :
#        """
#        Crank-Nicholson solver for marching through time
#        """
#        Cur_prev, Cur, Cur_nxt = Cur_vec[0], Cur_vec[1], Cur_vec[2]

#        maxIters = 10
#        tol      = 1e-4

#        Nx = self.N+self.Na+self.Nc
#        Nz = self.Na + self.Nc + self.N + self.Na + self.Nc

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

##            self.Input = Cur_nxt
##            jmat_num = compute_deriv( self.dae_system_num, numpy.concatenate( (x_nxt[:,idx], z_nxt[:,idx]) ) )

##            fx_num = jmat_num[:self.num_diff_vars,:self.num_diff_vars]
##            fz_num = jmat_num[:self.num_diff_vars,self.num_diff_vars:]
##            gx_num = jmat_num[self.num_diff_vars:,:self.num_diff_vars]
##            gz_num = jmat_num[self.num_diff_vars:,self.num_diff_vars:]

##            F1x_num = -sparse.eye(len(x)) + delta_t/2. * fx_num
##            F1z_num = delta_t/2. * fz_num

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

##        jm1_sp = sps.csr_matrix(jmat)
##        jm2_sp = sps.csr_matrix(jmat_num)

##        fig, ax = plt.subplots(1,2)
##        ax[0].spy( jm1_sp )
##        ax[1].spy( jm2_sp )
##        plt.show()

#        return x_nxtf, z_nxtf, newtonStats





