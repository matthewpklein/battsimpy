import numpy
import numpy.linalg
import scipy.integrate as sci_int
import scipy.linalg

from matplotlib import pyplot as plt

from assimulo.solvers import IDA
from assimulo.problem import Implicit_Problem

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

    def __init__(self, N, Ac, y0, yd0, name ) :

        Implicit_Problem.__init__(self,y0=y0,yd0=yd0,name=name)

        self.Ac = Ac
        self.T = 298.15

        La = 65.0
        Ls = 25.0
        Lc = 55.0
        Lt = (La+Ls+Lc)
        X = Lt*1e-6 # [m]

        Ns = int(N*(Ls/Lt))
        Na = int(N*(La/Lt))
        Nc = N - Ns - Na

        print 'Na, Nc:', Na, Nc

        self.N = N
        self.X = X

        self.x_e  = numpy.linspace( 0.0, X, N+1 )
        self.x_m  = numpy.array( [ 0.5*(self.x_e[i+1]+self.x_e[i]) for i in range(N) ], dtype='d'  )
        self.vols = numpy.array( [ (self.x_e[i+1] - self.x_e[i]) for i in range(N)], dtype='d' )

        ### Diffusivity
        self.La, self.Ls, self.Lc = La, Ls, Lc
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

        self.K_m = numpy.diag( self.k_m )

        self.pe0_ind = self.Na+self.Ns+self.Nc-3

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

        Bap = [ asa*F*v for asa,v in zip(as_a,self.vols[:Na]) ]
        Bsp = [   0.0 for i   in range(Ns) ]
        Bcp = [ asc*F*v for asc,v in zip(as_c,self.vols[-Nc:]) ]
#        Bap = [ asa*F for asa in as_a ]
#        Bsp = [   0.0 for i   in range(Ns) ]
#        Bcp = [ asc*F for asc in as_c ]

        self.B2_pe = numpy.diag( numpy.array(Bap+Bsp+Bcp, dtype='d') )

    def set_j_vec( self, I_app ) :
        i_app = I_app / self.Ac
        j_in_a  =  i_app / ( self.La*self.as_a_mean*self.F ) 
        j_in_c  = -i_app / ( self.Lc*self.as_c_mean*self.F ) 
        print 'i_app :', i_app
        print 'j_in_a:', j_in_a
        print 'j_in_c:', j_in_c
        # Set the input j
        ja = [ j_in_a for i in range(self.Na) ]
        js = [  0.0  for i in range(self.Ns) ]
        jc = [ j_in_c for i in range(self.Nc) ]

        self.j = numpy.array( ja + js + jc )

    ## Define c_e functions
    def build_Ace_mat( self, c ) :

        D_eff = self.Diff_ce( c )

        A = self.K_m.dot( flux_mat_builder( self.N, self.x_m, self.vols, D_eff ) )

        return A

    def Diff_ce( self, c ) :

        T = self.T

        D_ce = 1e-4 * 10.0**( -4.43 - (54./(T-229.-5e-3*c)) - (0.22e-3*c) )  ## Torchio (LIONSIMBA) ECS paper

        #1e-10*numpy.ones_like(c)# 

        D_mid = D_ce * self.eps_eff

        if type(c) == float :
            D_edge = D_mid
        else :
            D_edge = mid_to_edge( D_mid, self.x_e )

        return D_edge

    ## Define phi_e functions
    def build_Ape_mat( self, c ) :

        k_eff = self.kapp_ce( c )

        A = flux_mat_builder( self.N, self.x_m, numpy.ones_like(self.vols), k_eff )

        A[-1,-1] = 2*A[-1,-1]

        return A

    def build_Bpe_mat( self, c ) :

        gam = 2.*(1.-self.t_plus)*self.R_gas / self.F

        k_eff = self.kapp_ce( c ) #0.1*numpy.ones_like(c)#

        c_edge = mid_to_edge( c, self.x_e )

        B1 = flux_mat_builder( self.N, self.x_m, numpy.ones_like(self.vols), k_eff*self.T*gam/c_edge )

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

        ce     = y[ :N]
        c_dots = yd[:N]

        phi = y[N:]

        A_ce = self.build_Ace_mat( ce )
        A_pe = self.build_Ape_mat( ce )
        B_pe = self.build_Bpe_mat( ce )

        r1 = c_dots - ( ((A_ce.dot(ce)).flatten() + (self.B_ce.dot(self.j)).flatten()) )

        r2 = A_pe.dot(phi).flatten() - B_pe.dot(ce).flatten() + self.B2_pe.dot(self.j).flatten()

        res_out = numpy.concatenate( [r1, r2] )

        return res_out

    def jac( self, c, t, y, yd ) :

        ce     = y[ :N]
        c_dots = yd[:N]

        phi = y[N:]

        A_ce = self.build_Ace_mat( ce )
        A_pe = self.build_Ape_mat( ce )
        B_pe = self.build_Bpe_mat( ce )

        A_dots = numpy.diag( [1*c for i in range(self.N)] )
        j_c = A_dots - A_ce

        j = scipy.linalg.block_diag( j_c, A_pe )

        j[self.N:,:self.N] = -B_pe

        return j


### Mesh
N = 80
#X = 165e-6 # [m]

cell_coated_area = 1.0 # [m^2]
I_app = 200.0 #300.0 # A

### Initial conditions
c_init = 1000.0 # [mol/m^3]
c_centered = c_init*numpy.ones( N, dtype='d' )#numpy.linspace(1500, 500, N) #
p_init = 0.0 # [V]
p_centered = p_init*numpy.ones( N, dtype='d' )

#The initial conditons
y0 = numpy.concatenate( [c_centered, p_centered] ) #Initial conditions
yd0 = [0.0 for i in range(2*N)] #Initial conditions

#Create an Assimulo implicit problem
imp_mod = MyProblem(N,cell_coated_area,y0,yd0,'Example using an analytic Jacobian')

#Sets the options to the problem
#imp_mod.jac = jac #Sets the jacobian
imp_mod.algvar = [1.0 for i in range(N)] + [0.0 for i in range(N)] #Set the algebraic components

#Create an Assimulo implicit solver (IDA)
imp_sim = IDA(imp_mod) #Create a IDA solver

#Sets the paramters
imp_sim.atol = 1e-5 #Default 1e-6
imp_sim.rtol = 1e-5 #Default 1e-6
imp_sim.suppress_alg = True #Suppres the algebraic variables on the error test


#Simulate

#res_test = imp_mod.res( 0.0, y0, yd0 )
#jac_test = imp_mod.jac( 2, 0.0, y0, yd0 )

#Let Sundials find consistent initial conditions by use of 'IDA_YA_YDP_INIT'
imp_mod.set_j_vec( I_app )
imp_sim.make_consistent('IDA_YA_YDP_INIT')
# Sim step 1
t1, y1, yd1 = imp_sim.simulate(10,100) 

#imp_mod.set_j_vec( I_app/10. )
#imp_sim.make_consistent('IDA_YA_YDP_INIT')
## Sim step 1
#ta, ya, yda = imp_sim.simulate(10.05,100) 

#imp_mod.set_j_vec( I_app/100. )
#imp_sim.make_consistent('IDA_YA_YDP_INIT')
## Sim step 1
#tb, yb, ydb = imp_sim.simulate(10.1,100) 

imp_mod.set_j_vec( 0.0 )
imp_sim.make_consistent('IDA_YA_YDP_INIT')
# Sim step 1
t2, y2, yd2 = imp_sim.simulate(1100,100) 

c_avg_0 = numpy.mean( imp_mod.eps_m*y0[:N] )
c_avg_f = numpy.mean( imp_mod.eps_m*y2[-1,:N] )

print c_avg_0
print c_avg_f

#Plot

# Plot through space
f, ax = plt.subplots(2,2)
ax[0,0].plot(imp_mod.x_m*1e6,y1.T[:N]) #Plot the solution
ax[0,1].plot(imp_mod.x_m*1e6,y1.T[N:]) #Plot the solution

ax[0,0].set_title('t1 c')
ax[0,0].set_xlabel('Cell Thickness [$\mu$m]')
ax[0,0].set_ylabel('E-lyte Conc. [mol/m$^3$]')
ax[0,1].set_title('t1 p')
ax[0,1].set_xlabel('Cell Thickness [$\mu$m]')
ax[0,1].set_ylabel('E-lyte Potential [V]')

ax[1,0].plot(imp_mod.x_m*1e6,y2.T[:N]) #Plot the solution
ax[1,1].plot(imp_mod.x_m*1e6,y2.T[N:]) #Plot the solution

ax[1,0].set_title('t2 c')
ax[1,0].set_xlabel('Cell Thickness [$\mu$m]')
ax[1,0].set_ylabel('E-lyte Conc. [mol/m$^3$]')
ax[1,1].set_title('t2 p')
ax[1,1].set_xlabel('Cell Thickness [$\mu$m]')
ax[1,1].set_ylabel('E-lyte Potential [V]')

plt.tight_layout()

# Plot through time
f, ax = plt.subplots(1,2)
ax[0].plot(t1,y1[:,:N]) #Plot the solution
ax[1].plot(t1,y1[:,N:]) #Plot the solution

ax[0].plot(t2,y2[:,:N]) #Plot the solution
ax[1].plot(t2,y2[:,N:]) #Plot the solution

ax[0].set_ylabel('E-lyte Conc. [mol/m$^3$]')
ax[0].set_xlabel('Time [s]')
ax[1].set_ylabel('E-lyte Potential [V]')
ax[1].set_xlabel('Time [s]')

plt.tight_layout()

plt.show()
