import numpy
from matplotlib import pyplot as plt

nmc_cby25_111 = numpy.loadtxt( '/home/m_klein/Projects/battsimpy/data/Model_v1/Model_Pars/solid/thermodynamics/2012Wu_NMC111_Cby25_dchg.csv', delimiter=',' )
nmc_cby6_523  = numpy.loadtxt( '/home/m_klein/Projects/battsimpy/data/Model_v1/Model_Pars/solid/thermodynamics/2012Yang_NMC532_Cby6_dchg.csv', delimiter=',' )

nmc_rest_523  = numpy.loadtxt( '/home/m_klein/Projects/battsimpy/data/Model_v1/Model_Pars/solid/thermodynamics/2012Yang_523NMC_dchg_restOCV.csv', delimiter=',' )

plt.figure(1)

plt.plot( nmc_cby25_111[:,0], nmc_cby25_111[:,1], label='523_Cby25' )

plt.plot( nmc_cby6_523[:,0]+(.118-.045), nmc_cby6_523[:,1], label='111_Cby6' )

plt.plot( nmc_rest_523[:,0], nmc_rest_523[:,1], label='111_rest' )

plt.legend()

plt.show()
