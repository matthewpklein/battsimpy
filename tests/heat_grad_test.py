import numpy
from matplotlib import pyplot as plt

def grad_mat( N, x ) :

    G = numpy.zeros( [N,N] )
    for i in range(1,N-1) :
        G[i,[i-1, i+1]] = [ -1./(x[i+1]-x[i-1]), 1./(x[i+1]-x[i-1]) ]
    G[0,[0,1]] = [-1./(x[1]-x[0]),1./(x[1]-x[0])]
    G[-1,[-2,-1]] = [-1./(x[-1]-x[-2]),1./(x[-1]-x[-2])]

    return G

### Control volumes and node points (mid node points and edge node points)
Ns = 10
Na = 20
Nc = 20

N = Na + Ns + Nc
X = 1e-6*(55+25+65)

x_e  = numpy.linspace( 0.0, X, N+1 )
x_m  = numpy.array( [ 0.5*(x_e[i+1]+x_e[i]) for i in range(N) ], dtype='d'  )
vols = numpy.array( [ (x_e[i+1] - x_e[i]) for i in range(N)], dtype='d' )


# Useful sub-meshes for the phi_s functions
x_m_a = x_m[:Na]
x_m_c = x_m[-Nc:]
x_e_a = x_e[:Na+1]
x_e_c = x_e[-Nc-1:]

vols_a = vols[:Na]
vols_c = vols[-Nc:]

k=0.1
K = numpy.diag( numpy.ones(N)*k )

G = grad_mat( N, x_m )

phi = numpy.linspace( 0.2, 0., N )

phi = phi**2 / 0.1

dphi = numpy.gradient(phi)/numpy.gradient(x_m)
eq0 = sum( k*(dphi**2)*vols )
eq1 = sum( k*(G.dot(phi))*(G.dot(phi))*vols )
eq2 = vols.dot( k*(G.dot(phi))*(G.dot(phi)) )
eq3 = (vols.dot( k*numpy.diag(G.dot(phi)).dot(G) )).dot(phi)

print (vols.dot( k*numpy.diag(G.dot(phi)).dot(G) ))

print eq0, eq1, eq2, eq3


plt.figure()
plt.plot( x_m, G.dot(phi) )
plt.show()

