import numpy
import numpy.linalg
import scipy.integrate as sci_int

from matplotlib import pyplot as plt

from assimulo.solvers import CVode
from assimulo.problem import Explicit_Problem

## Helper functions
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

class MyProblem( Explicit_Problem ) :
    def __init__(self, N, X, y0, name ) :
        Explicit_Problem.__init__(self,y0=y0,name=name)
#        self.T = 298.15

        self.N = N
        self.X = X

        self.x_e  = numpy.linspace( 0.0, X, N+1 )
        self.x_m  = numpy.array( [ 0.5*(self.x_e[i+1]+self.x_e[i]) for i in range(N) ], dtype='d'  )
        self.vols = numpy.array( [ (self.x_e[i+1] - self.x_e[i]) for i in range(N)], dtype='d' )

        ### Diffusivity
        Ns = int(N/8.)
        Na = int(N/3.)
        Nc = N - Ns - Na
        self.La, self.Ls, self.Lc = Na*X/N, Ns*X/N, Nc*X/N
        self.Na, self.Ns, self.Nc = Na, Ns, Nc
        eps_a = 0.3
        eps_s = 0.5
        eps_c = 0.25
        ba, bs, bc = 0.8, 0.5, 0.5

        eps_a_vec = [ eps_a for i in range(Na) ] # list( eps_a + eps_a/2.*numpy.sin(numpy.linspace(0.,Na/4,Na)) ) # list(eps_a + eps_a*numpy.random.randn(Na)/5.) # 
        eps_s_vec = [ eps_s for i in range(Ns) ]
        eps_c_vec = [ eps_c for i in range(Nc) ] # list( eps_c + eps_c/2.*numpy.sin(numpy.linspace(0.,Nc/4,Nc)) ) # list(eps_c + eps_c*numpy.random.randn(Nc)/5.) # 

        self.eps_m   = numpy.array( eps_a_vec + eps_s_vec + eps_c_vec )
        self.K_m     = 1./self.eps_m
        self.eps_mb  = numpy.array( [ ea**ba for ea in eps_a_vec ] + [ es**bs for es in eps_s_vec ] + [ ec**bc for ec in eps_c_vec ] )
        self.eps_eff = numpy.array( [ ea**(1.+ba) for ea in eps_a_vec ] + [ es**(1.+bs) for es in eps_s_vec ] + [ ec**(1.+bc) for ec in eps_c_vec ] )

        t_plus = 0.4
        F = 96485.0

        Rp_c = 6.5e-6
        Rp_a = 12.0e-6

        as_c = 3.*eps_c/Rp_c
        as_a = 3.*eps_a/Rp_a
        self.as_c = as_c
        self.as_a = as_a

        Ba = [ (1.-t_plus)*as_a/eps_a for i in range(Na) ]
        Bs = [  0.0                     for i in range(Ns) ]
        Bc = [ (1.-t_plus)*as_c/eps_c for i in range(Nc) ]

        self.B = numpy.diag( Ba+Bs+Bc )

    def set_j_vec( self, j_in ) :
        # Set the input j
        ja = [  j_in for i in range(self.Na) ]
        js = [  0.0  for i in range(self.Ns) ]
        jc = [ -j_in * self.as_a*self.La / (self.as_c*self.Lc) for i in range(self.Nc) ]

        self.j = numpy.array( ja + js + jc )

    ## c_e functions
    def build_Ace_mat( self, c ) :

        D_eff = self.Diff_ce( c )

        A = numpy.diag( self.K_m ).dot( flux_mat_builder( self.N, self.x_m, self.vols, D_eff ) )

        return A

    def Diff_ce( self, c ) :

        T = 298.15 #self.T

        D_ce = 1e-4 * 10.0**( -4.43 - (54./(T-229.-5e-3*c)) - (0.22e-3*c) )  ## Torchio (LIONSIMBA) ECS paper

        D_mid = D_ce * self.eps_eff

        if type(c) == float :
            D_edge = D_mid
        else :
            D_edge = mid_to_edge( D_mid, self.x_e )

        return D_edge

    def jac( self, t, c ) :

        return self.build_Ace_mat( c )

    ## Derivative output
    def rhs( self, t, c ) :

        A = self.build_Ace_mat( c )

        c_dots = (A.dot(c)).flatten() + (self.B.dot(self.j)).flatten()

        return c_dots


### Mesh
N = 60
X = 165e-6 # [m]

### Initial conditions
c_init = 1000.0 # [mol/m^3]
c_centered = c_init*numpy.ones( N, dtype='d' )

exp_mod = MyProblem( N, X, c_centered, 'ce only model, explicit CVode' )
#exp_mod = MyProblem( N, X, numpy.linspace(c_init-c_init/5.,c_init+c_init/5.,N), 'ce only model, explicit CVode' )

# Set the ODE solver
exp_sim = CVode(exp_mod) #Create a CVode solver

#Set the parameters
exp_sim.iter  = 'Newton' #Default 'FixedPoint'
exp_sim.discr = 'BDF' #Default 'Adams'
exp_sim.atol = 1e-5 #Default 1e-6
exp_sim.rtol = 1e-5 #Default 1e-6

#Simulate
exp_mod.set_j_vec( 1.e-4 )
t1, y1 = exp_sim.simulate(100, 100)

exp_mod.set_j_vec( 0.0 )
t2, y2 = exp_sim.simulate(200, 100)

#exp_mod.set_j_vec( 0.0 )
#t3, y3 = exp_sim.simulate(10000, 200)

#Plot
#if with_plots:
f, ax = plt.subplots(1,2)
ax[0].plot(exp_mod.x_m*1e6,y1.T) #Plot the solution
ax[1].plot(exp_mod.x_m*1e6,y2.T) #Plot the solution
#ax[2].plot(exp_mod.x_m*1e6,y3.T) #Plot the solution
#plt.xlabel('Cell Thickness [$\mu$m]')
#plt.ylabel('State')
#plt.title(exp_mod.name)
#ax[0].ylim([0,3000])
f, ax = plt.subplots(1,2)
ax[0].plot(t1,y1) #Plot the solution
ax[1].plot(t2,y2) #Plot the solution
#ax[2].plot(t3,y3) #Plot the solution

plt.show()


