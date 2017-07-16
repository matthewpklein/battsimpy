import numpy
import numpy.linalg
import scipy.integrate as sci_int

from matplotlib import pyplot as plt

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
def build_A_mat( N, x_mid, vols, D ) :

    return flux_mat_builder( N, x_mid, vols, D )


def jac( t, y, N, x_mid, vols, D ) :

    return build_A_mat( N, x_mid, vols, D )


## Derivative output
def dots( t, c, A, B, j ) :

    print t, j

    c_dots = (A.dot(c)).flatten() + (B.dot(j)).flatten()

    return c_dots


### Mesh
N = 15
X = 5.0e-6 # [m]
#x_edge = numpy.linspace( 0.0, X, N+1 )
x_edge = nonlinspace( X, 0.85, N+1 )


x_mid  = numpy.array( [ 0.5*(x_edge[i+1]+x_edge[i]) for i in range(N) ], dtype='d'  )

### Plot of the x vectors
#plt.figure(1)
#plt.plot( x_edge, numpy.ones_like(x_edge), '-*r', label='edge' )
#plt.plot( x_mid, 0.98*numpy.ones_like(x_mid), '-sg', label='mid' )
#plt.legend()
#plt.ylim([0.8,1.2])

vols = numpy.array( [ 1./3.*(x_edge[i+1]**3 - x_edge[i]**3) for i in range(N)], dtype='d' )

#plt.figure(2)
#plt.plot( vols     , '*-', label='vols' )
#plt.legend()

### Initial conditions
c_init = 1000.0 # [mol/m^3]

c_edge     = c_init*numpy.ones_like( x_edge, dtype='d' )
c_centered = c_init*numpy.ones_like( x_mid , dtype='d' )

Ds = 1e-14
D_mid  = Ds*numpy.ones_like( x_mid  )
D_edge = Ds*numpy.ones_like( x_edge )

### System matrix build
A = build_A_mat( N, x_mid, vols, D_edge*(x_edge**2) )

B = numpy.array( [0. for i in range(N-1)]+[x_edge[-1]**2/vols[-1]], dtype='d' )

print 'A:', '\n', A , '\n', '', '\n', B

### Simulation
sim_on = 1
plt_on = 1

if sim_on :
    j = 1e-6

    tf = 100.
    NT = int(5*tf)
    t = numpy.linspace( 0.,tf, NT )

    j_in = j

    ### integrate.ode class implementation -> more control over solver, etc. ###
    ode15s = sci_int.ode(dots,jac=jac)
    ode15s.set_integrator('vode', method='bdf')
    ode15s.set_f_params(A, B, j_in)
    ode15s.set_jac_params( N, x_mid, vols, D_edge*(x_edge**2) )

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
    ode15s.set_f_params(A, B, j_in)
    ode15s.set_jac_params( N, x_mid, vols, D_edge*(x_edge**2) )
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


    j_in = -j
    tf = 100.
    NT = int(5*tf)
    t = numpy.linspace( 0.,tf, NT )
    ode15s = sci_int.ode(dots,jac=jac)
    ode15s.set_integrator('vode', method='bdf')
    ode15s.set_f_params(A, B, j_in)
    ode15s.set_jac_params( N, x_mid, vols, D_edge*(x_edge**2) )
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



    j_in = 0.0*j
    tf = 2000.
    NT = int(0.5*tf)
    t = numpy.linspace( 0.,tf, NT )
    ode15s = sci_int.ode(dots,jac=jac)
    ode15s.set_integrator('vode', method='bdf')
    ode15s.set_f_params(A, B, j_in)
    ode15s.set_jac_params( N, x_mid, vols, D_edge*(x_edge**2) )
    ode15s.set_initial_value(x2[-1,:], t[0])
    tode = numpy.zeros_like(t) # preallocate the time vector used to store the ode integrator time steps
    x = numpy.zeros( (NT,len(c_centered)), dtype='d' ) # preallocate the state output array
    x[0,:] = x2[-1,:]
    tode[0] = t[0]
    itime = 1 # initialize the time step index
    while ode15s.successful() and itime < NT :
        delta_t = t[itime] - t[itime-1]
        ode15s.integrate( ode15s.t + delta_t )

        tode[itime] = ode15s.t
        x[itime,:]  = ode15s.y
        itime += 1

    x3 = x[:itime,:]
    t3 = tode[:itime]



    if plt_on :
        
        print x0[0,:]
        print x3[-1, :]

        print x0[0,:] - x3[-1,:]

        #print x
        g = 100
        plt_inds = [ i*g for i in range(NT/g) ]
        plt.figure(5)
        plt.plot( x_mid*1e6, x0.T/c_init )
        #for ip in plt_inds :
        #    plt.plot( x_mid, x[ip,:] )
        plt.figure(6)
        plt.plot( x_mid*1e6, x0[ 0,:]/c_init, '-s', label='init' )
        plt.plot( x_mid*1e6, x3[-1,:]/c_init, '-d', label='final' )
        plt.ylim( [0.9, 1.1] )
        plt.legend()

        f, ax = plt.subplots(1,4)
        ax[0].plot(x_mid*1e6, x0.T/c_init)
        ax[1].plot(x_mid*1e6, x1.T/c_init)
        ax[2].plot(x_mid*1e6, x2.T/c_init)
        ax[3].plot(x_mid*1e6, x3.T/c_init)

        plt.show()
