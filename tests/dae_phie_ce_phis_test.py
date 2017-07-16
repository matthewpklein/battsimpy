import numpy
import numpy.linalg
import scipy.linalg

from matplotlib import pyplot as plt

from assimulo.solvers import IDA
from assimulo.problem import Implicit_Problem

from scipy.sparse.linalg import spsolve as sparseSolve
from scipy.sparse import csr_matrix as sparseMat
import scipy.sparse as sparse
import math
from copy import deepcopy

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

    def __init__(self, Na, Ns, Nc, X, Ac, y0, yd0, name ) :

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

        Rp_c = 6.5e-6
        Rp_a = 12.0e-6

        as_c = 3.*numpy.array(eps_c_vec, dtype='d')/Rp_c
        as_a = 3.*numpy.array(eps_a_vec, dtype='d')/Rp_a
        self.as_c = as_c
        self.as_a = as_a

        self.as_a_mean = 1./self.La*sum( [ asa*v for asa,v in zip(as_a, self.vols[:Na]) ] )
        self.as_c_mean = 1./self.Lc*sum( [ asc*v for asc,v in zip(as_c, self.vols[-Nc:]) ] )

        print 'asa diff', self.as_a_mean - as_a[0]
        print 'asc diff', self.as_c_mean - as_c[0]

        Ba = [ (1.-t_plus)*asa/ea for ea, asa in zip(eps_a_vec,as_a) ]
        Bs = [  0.0                for i in range(Ns) ]
        Bc = [ (1.-t_plus)*asc/ec for ec, asc in zip(eps_c_vec,as_c) ]

        self.B_ce = numpy.diag( numpy.array(Ba+Bs+Bc, dtype='d') )

#        Bap = [ asa*F*dxa for asa,dxa in zip(as_a,self.vols_a) ]
#        Bsp = [   0.0 for i   in range(Ns) ]
#        Bcp = [ asc*F*dxc for asc,dxc in zip(as_c,self.vols_c) ]
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


    def set_j_vec( self, I_app ) :
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

#        A = flux_mat_builder( self.N, self.x_m, numpy.ones_like(self.vols), k_eff )
        A = flux_mat_builder( self.N, self.x_m, self.vols, k_eff )

        A[-1,-1] = 2*A[-1,-1]

        return A

    def build_Bpe_mat( self, c ) :

        gam = 2.*(1.-self.t_plus)*self.R_gas / self.F

        k_eff = self.kapp_ce( c )

#        B1 = numpy.diag( 1./c ).dot( flux_mat_builder( self.N, self.x_m, numpy.ones_like(self.vols), k_eff*self.T*gam ) )
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


    ## Define system equations
    def res( self, t, y, yd ) :

        ce     = y[ :self.N]
        c_dots = yd[:self.N]

        phi = y[self.N:2*self.N]

        phi_s_a = y[2*self.N:-self.Nc]
        phi_s_c = y[-self.Nc:]

        A_ce = self.build_Ace_mat( ce )
        A_pe = self.build_Ape_mat( ce )
        B_pe = self.build_Bpe_mat( ce )

        r1 = c_dots - ( ((A_ce.dot(ce)).flatten() + (self.B_ce.dot(self.j)).flatten()) )

        r2 = A_pe.dot(phi).flatten() - B_pe.dot(ce).flatten() + self.B2_pe.dot(self.j).flatten()

        r3 = self.A_ps_a.dot(phi_s_a).flatten() - self.B_ps_a.dot(self.j_a).flatten() - self.B2_ps_a*self.i_app
        r4 = self.A_ps_c.dot(phi_s_c).flatten() - self.B_ps_c.dot(self.j_c).flatten() + self.B2_ps_c*self.i_app

        res_out = numpy.concatenate( [r1, r2, r3, r4] )

#        res_out = numpy.concatenate( [r1, r2] )

        return res_out

    def jac( self, c, t, y, yd ) :

        ce = y[:self.N]

        A_ce = self.build_Ace_mat( ce )
        A_pe = self.build_Ape_mat( ce )
        B_pe = self.build_Bpe_mat( ce )

        A_dots = numpy.diag( [1*c for i in range(self.N)] )
        j_c = A_dots - A_ce

        j = scipy.linalg.block_diag( j_c, A_pe, self.A_ps_a, self.A_ps_c )

        j[self.N:2*self.N,:self.N] = -B_pe

        return j


#    def dae_system( self, x, z, Input, get_mats=0 ) :

#        self.set_j_vec(Input)

#        ce = x

#        phi = z[:self.N]

#        phi_s_a = z[self.N:-self.Nc]
#        phi_s_c = z[-self.Nc:]

#        A_ce = self.build_Ace_mat( ce )
#        A_pe = self.build_Ape_mat( ce )
#        B_pe = self.build_Bpe_mat( ce )

#        r1 = ( ((A_ce.dot(ce)).flatten() + (self.B_ce.dot(self.j)).flatten()) )

#        r2 = A_pe.dot(phi).flatten() - B_pe.dot(ce).flatten() + self.B2_pe.dot(self.j).flatten()

#        r3 = self.A_ps_a.dot(phi_s_a).flatten() - self.B_ps_a.dot(self.j_a).flatten() - self.B2_ps_a*self.i_app
#        r4 = self.A_ps_c.dot(phi_s_c).flatten() - self.B_ps_c.dot(self.j_c).flatten() + self.B2_ps_c*self.i_app

#        if get_mats :
#            res_out = r1, numpy.concatenate( [r2, r3, r4] ), { 'A_ce':A_ce, 'A_pe':A_pe, 'B_pe':B_pe }
#        else :
#            res_out = r1, numpy.concatenate( [r2, r3, r4] )

#        return res_out

#    def jac_system( self, mats ) :

#        A_ce = mats['A_ce'] #self.build_Ace_mat( ce )
#        A_pe = mats['A_pe'] #self.build_Ape_mat( ce )
#        B_pe = mats['B_pe'] #self.build_Bpe_mat( ce )

##        A_dots = numpy.diag( [1*cc for i in range(self.N)] )
##        j_c = A_dots - A_ce

##        j = scipy.linalg.block_diag( A_ce, A_pe, self.A_ps_a, self.A_ps_c )

##        j[self.N:2*self.N,:self.N] = -B_pe

#        fx =  A_ce
#        fz =  numpy.zeros( [self.N, self.N+self.Na+self.Nc] )
#        gx =  numpy.zeros( [self.N+self.Na+self.Nc, self.N] )
#        gx[:N,:N] = -B_pe
#        gz =  scipy.linalg.block_diag( A_pe, self.A_ps_a, self.A_ps_c )

#        return fx, fz, gx, gz

#    def cn_solver( self, x, z, Cur_vec, delta_t ) :
#        """
#        Crank-Nicholson solver for marching through time
#        """
#        Cur_prev, Cur, Cur_nxt = Cur_vec[0], Cur_vec[1], Cur_vec[2]

#        maxIters = 20
#        tol      = 1e-5

#        Nx = self.N
#        Nz = self.N + self.Na + self.Nc

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


### Mesh
N = 60
Ns = int(N/8.)
Na = int(N/3.)
Nc = N - Ns - Na

X = 165e-6 # [m]

cell_coated_area = 1.0 # [m^2]
I_app = 10.0 # A
#i_app = I_app / cell_coated_area # current density, [A/m^2]

### Initial conditions
c_init = 1000.0 # [mol/m^3]
c_centered = c_init*numpy.ones( N, dtype='d' )
p_init = 0.0 # [V]
p_centered = p_init*numpy.ones( N, dtype='d' )

pa_init = 0.0 # [V]
pa_centered = pa_init*numpy.ones( Na, dtype='d' )
pc_init = 0.0 # [V]
pc_centered = pc_init*numpy.ones( Nc, dtype='d' )

#The initial conditons
y0 = numpy.concatenate( [c_centered, p_centered, pa_centered, pc_centered] ) #Initial conditions
yd0 = [0.0 for i in range(2*N+Na+Nc)] #Initial conditions
#yd0 = [0.0 for i in range(2*N)] #Initial conditions

#Create an Assimulo implicit problem
imp_mod = MyProblem(Na,Ns,Nc,X,cell_coated_area,y0,yd0,'Example using an analytic Jacobian')

imp_mod.set_j_vec( I_app )

#delta_t = 1.0
#tf = 10.
#time = [ i*delta_t for i in range(int(tf/delta_t)+1) ]

#print time

#x_out = numpy.zeros( [imp_mod.N, len(time)] )
#z_out = numpy.zeros( [imp_mod.N+imp_mod.Na+imp_mod.Nc, len(time)] )

#x_out[:,0] = c_centered
#z_out[:,0] = numpy.concatenate( [p_centered, pa_centered, pc_centered] )

#for it, t in enumerate(time[1:]) :

#    if it == 0 :
#        Cur_vec = [ 0.0, 0.0, I_app ]
#    elif it == 1 :
#        Cur_vec = [ 0.0, I_app, I_app ]
#    else :
#        Cur_vec = [ I_app, I_app, I_app ]
#        
#    x_out[:,it+1], z_out[:,it+1], newtonStats = imp_mod.cn_solver( x_out[:,it], z_out[:,it], Cur_vec, delta_t )


#f, ax = plt.subplots(1,3)
#ax[0].plot( imp_mod.x_m, x_out )

#ax[1].plot( imp_mod.x_m, z_out[:imp_mod.N,:-1] )

#ax[2].plot( imp_mod.x_m_a, z_out[imp_mod.N:imp_mod.N+imp_mod.Na,:-1] )
#ax[2].plot( imp_mod.x_m_c, z_out[-imp_mod.Nc:,:-1] )
#plt.show()

#print z_out

#Sets the options to the problem
#imp_mod.jac = jac #Sets the jacobian
imp_mod.algvar = [1.0 for i in range(N)] + [0.0 for i in range(N+Na+Nc)] #Set the algebraic components
#imp_mod.algvar = [1.0 for i in range(N)] + [0.0 for i in range(N)] #Set the algebraic components

#Create an Assimulo implicit solver (IDA)
imp_sim = IDA(imp_mod) #Create a IDA solver

#Sets the paramters
imp_sim.atol = 1e-5 #Default 1e-6
imp_sim.rtol = 1e-5 #Default 1e-6
imp_sim.suppress_alg = True #Suppres the algebraic variables on the error test


### Simulate
imp_mod.set_j_vec( I_app )

#res_test = imp_mod.res( 0.0, y0, yd0 )
#jac_test = imp_mod.jac( 2, 0.0, y0, yd0 )

#Let Sundials find consistent initial conditions by use of 'IDA_YA_YDP_INIT'
imp_sim.make_consistent('IDA_YA_YDP_INIT')
# Sim step 1
t1, y1, yd1 = imp_sim.simulate(100,100) 
#t1, y1, yd1 = imp_sim.simulate(1000,1000) 

imp_mod.set_j_vec( 0.0 )
imp_sim.make_consistent('IDA_YA_YDP_INIT')
# Sim step 1
t2, y2, yd2 = imp_sim.simulate(200,100) 


#Plot

# Plot through space
f, ax = plt.subplots(2,3)
ax[0,0].plot(imp_mod.x_m*1e6,y1.T[:N,:]) #Plot the solution
ax[0,1].plot(imp_mod.x_m*1e6,y1.T[N:2*N,:]) #Plot the solution

ax[0,2].plot(imp_mod.x_m_a*1e6,y1.T[2*N:2*N+Na,:]) #Plot the solution
ax[0,2].plot(imp_mod.x_m_c*1e6,y1.T[-Nc:,:]) #Plot the solution

ax[0,0].set_title('t1 c')
ax[0,0].set_xlabel('Cell Thickness [$\mu$m]')
ax[0,0].set_ylabel('E-lyte Conc. [mol/m$^3$]')
ax[0,1].set_title('t1 p')
ax[0,1].set_xlabel('Cell Thickness [$\mu$m]')
ax[0,1].set_ylabel('E-lyte Potential [V]')
ax[0,2].set_title('t1 p solid')
ax[0,2].set_xlabel('Cell Thickness [$\mu$m]')
ax[0,2].set_ylabel('Solid Potential [V]')

ax[1,0].plot(imp_mod.x_m*1e6,y2.T[:N,:]) #Plot the solution
ax[1,1].plot(imp_mod.x_m*1e6,y2.T[N:2*N,:]) #Plot the solution

ax[1,2].plot(imp_mod.x_m_a*1e6,y2.T[2*N:2*N+Na,:]) #Plot the solution
ax[1,2].plot(imp_mod.x_m_c*1e6,y2.T[-Nc:,:]) #Plot the solution

ax[1,0].set_title('t2 c')
ax[1,0].set_xlabel('Cell Thickness [$\mu$m]')
ax[1,0].set_ylabel('E-lyte Conc. [mol/m$^3$]')
ax[1,1].set_title('t2 p e-lyte')
ax[1,1].set_xlabel('Cell Thickness [$\mu$m]')
ax[1,1].set_ylabel('E-lyte Potential [V]')
ax[1,2].set_title('t2 p solid')
ax[1,2].set_xlabel('Cell Thickness [$\mu$m]')
ax[1,2].set_ylabel('Solid Potential [V]')

plt.tight_layout()

# Plot through time
f, ax = plt.subplots(1,3)
ax[0].plot(t1,y1[:,:N])
ax[1].plot(t1,y1[:,N:2*N])
ax[2].plot(t1,y1[:,2*N:2*N+Na]) 
ax[2].plot(t1,y1[:,-Nc:]) 

ax[0].plot(t2,y2[:,:N]) 
ax[1].plot(t2,y2[:,N:2*N]) 
ax[2].plot(t2,y2[:,2*N:2*N+Na]) 
ax[2].plot(t2,y2[:,-Nc:]) 

ax[0].set_ylabel('E-lyte Conc. [mol/m$^3$]')
ax[0].set_xlabel('Time [s]')
ax[1].set_ylabel('E-lyte Potential [V]')
ax[1].set_xlabel('Time [s]')

ax[2].set_ylabel('Solid Potential [V]')
ax[2].set_xlabel('Time [s]')

plt.tight_layout()

plt.show()
