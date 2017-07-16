import numpy
import numpy.linalg
import scipy.integrate as sci_int

from matplotlib import pyplot as plt

## Helper functions
def mid_to_edge( var_mid, x_edge ) :

    var_edge = numpy.array( [var_mid[0]] + [ var_mid[i]*var_mid[i+1]/( ((x_edge[i+1]-x_edge[i])/((x_edge[i+2]-x_edge[i+1])+(x_edge[i+1]-x_edge[i])))*var_mid[i+1] + (1- ((x_edge[i+1]-x_edge[i])/((x_edge[i+2]-x_edge[i+1])+(x_edge[i+1]-x_edge[i]))))*var_mid[i] ) for i in range(len(var_mid)-1) ] + [var_mid[-1]] )

    ve = [ 0. for i in range(len(var_mid)+1) ]
    for i in range(1,len(var_mid)) :

        dx1 = x_edge[i]   - x_edge[i-1]
        dx2 = x_edge[i+1] - x_edge[i]
        beta = dx1 / ( dx1 + dx2 )

        ve[i] = var_mid[i-1]*var_mid[i] / ( beta*var_mid[i] + (1.-beta)*var_mid[i-1] )

    ve[0] = var_mid[0]
    ve[-1] = var_mid[-1]

    plt.figure()
    plt.plot( x_edge, ve, '--sb' )
    plt.plot( x_edge, var_edge, '-*r' )
    plt.plot( x_mid, var_mid, '-dk' )
    plt.show()

    return var_edge

def flux_mat_builder( N, x_mid, vols, D ) :

    A = numpy.zeros([N,N], dtype='d')

    for i in range(1,N-1) :

        A[i,i-1] =  (1./vols[i]) * (D[i  ]) / (x_mid[i  ] - x_mid[i-1])
        A[i,i  ] = -(1./vols[i]) * (D[i  ]) / (x_mid[i  ] - x_mid[i-1]) - (1./vols[i]) * (D[i+1]) / (x_mid[i+1] - x_mid[i])
        A[i,i+1] =  (1./vols[i]) * (D[i+1]) / (x_mid[i+1] - x_mid[i  ])

    i=0
    A[0,0] = -(1./vols[i]) * (D[i+1]) / (x_mid[i+1] - x_mid[i])
    A[0,1] =  (1./vols[i]) * (D[i+1]) / (x_mid[i+1] - x_mid[i])

    i=N-1
    A[i,i-1] =  (1./vols[i]) * (D[i]) / (x_mid[i] - x_mid[i-1])
    A[i,i  ] = -(1./vols[i]) * (D[i]) / (x_mid[i] - x_mid[i-1])

    return A

## c_e functions
def build_Ace_mat( N, c_centered, x_mid, x_edge, vols, eps_mid ) :

    T = 298.15

    D_eff = Diff_ce( c_centered, T, eps_mid )

    A = flux_mat_builder( N, x_mid, vols, D_eff )

    return A

def Diff_ce( ce, T, eps_mid ) :

    D_ce = 1e-4 * 10.0**( -4.43 - (54./(T-229.-5e-3*ce)) - (0.22e-3*ce) )  ## Torchio (LIONSIMBA) ECS paper
#1e-10*numpy.ones_like(ce) #
    D_mid = D_ce * eps_mid

    if type(ce) == float :
        D_edge = D_mid
    else :
        D_edge = mid_to_edge( D_mid, x_edge )

    return D_edge

def jac( t, y, N, x_mid, x_edge, vols, eps_mid ) :

    return build_Ace_mat( N, y, x_mid, x_edge, vols, eps_mid )


## Derivative output
def dots( t, c, B, j ) :

    A = build_Ace_mat( N, c_centered, x_mid, x_edge, vols, eps_mid )

    c_dots = (A.dot(c)).flatten() + (B.dot(j)).flatten()

    return c_dots


### Mesh
N = 80
X = 165e-6 # [m]
x_edge = numpy.linspace( 0.0, X, N+1 )

x_mid  = numpy.array( [ 0.5*(x_edge[i+1]+x_edge[i]) for i in range(N) ], dtype='d'  )

### Plot of the x vectors
#plt.figure(1)
#plt.plot( x_edge, numpy.ones_like(x_edge), '-*r', label='edge' )
#plt.plot( x_mid, 0.98*numpy.ones_like(x_mid), '-sg', label='mid' )
#plt.legend()
#plt.ylim([0.8,1.2])

vols = numpy.array( [ (x_edge[i+1] - x_edge[i]) for i in range(N)], dtype='d' )

#plt.figure(2)
#plt.plot( vols     , '*-', label='vols' )
#plt.legend()

### Initial conditions
c_init = 1000.0 # [mol/m^3]

c_edge     = c_init*numpy.ones_like( x_edge, dtype='d' )
c_centered = c_init*numpy.ones_like( x_mid , dtype='d' )

### Diffusivity
Ns = int(N/8.)
Na = int(N/3.)
Nc = N - Ns - Na
La = Na*X/N
Lc = Nc*X/N

eps_a = 0.3
eps_s = 0.5
eps_c = 0.25
ba, bs, bc = 0.8, 0.5, 0.5

eps_a_vec = [ eps_a for i in range(Na) ] # list( eps_a + eps_a/2.*numpy.sin(numpy.linspace(0.,Na/4,Na)) ) # list(eps_a + eps_a*numpy.random.randn(Na)/5.) # 
eps_s_vec = [ eps_s for i in range(Ns) ]
eps_c_vec = [ eps_c for i in range(Nc) ] # list( eps_c + eps_c/2.*numpy.sin(numpy.linspace(0.,Nc/4,Nc)) ) # list(eps_c + eps_c*numpy.random.randn(Nc)/5.) # 

eps_mid = numpy.array( eps_a_vec + eps_s_vec + eps_c_vec )

eps_mid_brug = numpy.array( [ ea**ba for ea in eps_a_vec ] + [ es**bs for es in eps_s_vec ] + [ ec**bc for ec in eps_c_vec ] )

#D = Diff_ce(c_init,298.15,1.) #1.0e-10 # [m^2/s]

#D_mid  = D*eps_mid_brug
#D_edge = numpy.array( [D_mid[0]] + [ D_mid[i]*D_mid[i+1]/( ((x_edge[i+1]-x_edge[i])/((x_edge[i+2]-x_edge[i+1])+(x_edge[i+1]-x_edge[i])))*D_mid[i+1] + (1- ((x_edge[i+1]-x_edge[i])/((x_edge[i+2]-x_edge[i+1])+(x_edge[i+1]-x_edge[i]))))*D_mid[i] ) for i in range(len(x_mid)-1) ] + [D_mid[-1]] )

#D_edge1 = Diff_ce( c_centered, 298.15, eps_mid_brug )

#plt.figure(3)
#plt.plot( x_mid , D_mid , '*-', label='D_mid ' )
#plt.plot( x_edge, D_edge, 's-', label='D_edge' )
#plt.plot( x_edge, D_edge1, 'd--', label='D_edge1' )
#plt.legend()

#plt.figure(4)
#plt.plot( x_mid , eps_mid , '*-', label='eps_mid ' )
#plt.plot( x_mid, eps_mid_brug, 's-', label='eps_mid_brug' )
#plt.legend()

#plt.show()

t_plus = 0.4
F = 96485.0

### System matrix build
#A = build_A_mat( N, c_centered, x_mid, x_edge, vols, eps_mid_brug )

Rp_c = 6.5e-6
Rp_a = 12.0e-6

as_c = 3.*eps_c/Rp_c
as_a = 3.*eps_a/Rp_a

Ba = [ (1.-t_plus)/(eps_a)*as_a for i in range(Na) ]
Bs = [  0.0                     for i in range(Ns) ]
Bc = [ (1.-t_plus)/(eps_c)*as_c for i in range(Nc) ]

B = numpy.diag( Ba+Bs+Bc )

#print A, '\n', B

### Simulation
sim_on = 1
plt_on = 1

if sim_on :
    ja = [  1e-5 for i in range(Na) ]
    js = [  0.0  for i in range(Ns) ]
    jc = [ -ja[0] * as_a*La / (as_c*Lc) for i in range(Nc) ]

    j = numpy.array( ja + js + jc )

    tf = 100.
    NT = int(5*tf)
    t = numpy.linspace( 0.,tf, NT )

    j_in = j

    ### integrate.ode class implementation -> more control over solver, etc. ###
    ode15s = sci_int.ode(dots,jac=jac)
    ode15s.set_integrator('vode', method='bdf')
    ode15s.set_f_params(B, j_in)
    ode15s.set_jac_params(N, x_mid, x_edge, vols, eps_mid_brug)

    ode15s.set_initial_value(c_centered, t[0])
    tode = numpy.zeros_like(t) # preallocate the time vector used to store the ode integrator time steps
    x = numpy.zeros( (NT,len(c_centered)), dtype='d' ) # preallocate the state output array
    x[0,:] = c_centered
    tode[0] = t[0]
    itime = 1 # initialize the time step index
    while ode15s.successful() and itime < NT :
        delta_t = t[itime] - t[itime-1]
        ode15s.integrate( ode15s.t + delta_t )

        tode[itime] = ode15s.t
        x[itime,:]  = ode15s.y
        itime += 1

    x0 = x[:itime,:]
    t0 = tode[:itime]

    j_in = 0.*j
    ode15s = sci_int.ode(dots,jac=jac)
    ode15s.set_integrator('vode', method='bdf')
    ode15s.set_f_params(B, j_in)
    ode15s.set_jac_params(N, x_mid, x_edge, vols, eps_mid_brug)
    ode15s.set_initial_value(x0[-1,:], t[0])
    tode = numpy.zeros_like(t) # preallocate the time vector used to store the ode integrator time steps
    x = numpy.zeros( (NT,len(c_centered)), dtype='d' ) # preallocate the state output array
    x[0,:] = x0[-1,:]
    tode[0] = t[0]
    itime = 1 # initialize the time step index
    while ode15s.successful() and itime < NT :
        delta_t = t[itime] - t[itime-1]
        ode15s.integrate( ode15s.t + delta_t )

        tode[itime] = ode15s.t
        x[itime,:]  = ode15s.y
        itime += 1

    x1 = x[:itime,:]
    t1 = tode[:itime]

    j_in = 0.0*j
    tf = 2000.
    NT = int(0.5*tf)
    t = numpy.linspace( 0.,tf, NT )
    ode15s = sci_int.ode(dots,jac=jac)
    ode15s.set_integrator('vode', method='bdf')
    ode15s.set_f_params(B, j_in)
    ode15s.set_jac_params(N, x_mid, x_edge, vols, eps_mid_brug)
    ode15s.set_initial_value(x1[-1,:], t[0])
    tode = numpy.zeros_like(t) # preallocate the time vector used to store the ode integrator time steps
    x = numpy.zeros( (NT,len(c_centered)), dtype='d' ) # preallocate the state output array
    x[0,:] = x1[-1,:]
    tode[0] = t[0]
    itime = 1 # initialize the time step index
    while ode15s.successful() and itime < NT :
        delta_t = t[itime] - t[itime-1]
        ode15s.integrate( ode15s.t + delta_t )

        tode[itime] = ode15s.t
        x[itime,:]  = ode15s.y
        itime += 1

    x2 = x[:itime,:]
    t2 = tode[:itime]


    if plt_on :
        
        print x0[0,:]
        print x2[-1, :]

        print x0[0,:] - x2[-1,:]

        #print x
        g = 100
        plt_inds = [ i*g for i in range(NT/g) ]
        plt.figure(5)
        plt.plot( x_mid*1e6, x.T )
        #for ip in plt_inds :
        #    plt.plot( x_mid, x[ip,:] )
        plt.figure(6)
        plt.plot( x_mid*1e6, x0[ 0,:] , label='init' )
        plt.plot( x_mid*1e6, x2[-1,:], label='final' )
        plt.legend()

        f, ax = plt.subplots(1,3)
        ax[0].plot(x_mid*1e6, x0.T)
        ax[1].plot(x_mid*1e6, x1.T)
        ax[2].plot(x_mid*1e6, x2.T)

        plt.show()
