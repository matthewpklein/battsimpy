import numpy
from matplotlib import pyplot as plt

dir = '/home/mk-sim-linux/Battery_TempGrad/Python/batt_simulation/battsimpy/data/Model_nmc/Model_Pars/solid/thermodynamics/'

files = ['2012Yang_523NMC_dchg_restOCV.csv','2012Wu_NMC111_Cby25_dchg_2squeeze.csv']

dat = [ numpy.loadtxt( dir+f, delimiter=',' ) for f in files ]

x_intp = dat[0][:,0]

U2_intp = numpy.interp( x_intp-.01, dat[1][:,0], dat[1][:,1] )

w = 0.
plt.figure(1)
plt.plot( x_intp, dat[0][:,1], label='Yang' )
plt.plot( x_intp, U2_intp, label='Wu' )

Unew = numpy.mean( [(1.-w)*dat[0][:,1], (1.+w)*U2_intp], axis=0 )
plt.plot( x_intp, Unew, label='avg' )
plt.show()


numpy.savetxt( dir+'YangWuMix_NMC_20170607.csv', numpy.array([x_intp,Unew]).T, delimiter=',' )

