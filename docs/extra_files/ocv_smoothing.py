import batteqns
from matplotlib import pyplot as plt
import numpy

basedir = '/home/mk-sim-linux/Battery_TempGrad/Python/batt_simulation/battsimpy/'

#Uc_path = basedir+'data/Model_lfp/Model_Pars/solid/thermodynamics/2012Prada_LFP_U_dchg.csv'
#Ua_path = basedir+'data/Model_lfp/Model_Pars/solid/thermodynamics/Ua_cell2Fit_LFP_2012Prada.csv'

Uc_path = basedir+'data/Model_nmc/Model_Pars/solid/thermodynamics/2012Yang_523NMC_dchg_restOCV.csv'
Ua_path = basedir+'data/Model_nmc/Model_Pars/solid/thermodynamics/Ua_cell4Fit_NMC_2012Yang.csv'

raw_Ua = numpy.loadtxt( Ua_path, delimiter=',' )
raw_Uc = numpy.loadtxt( Uc_path, delimiter=',' )

Ua_intp, Uc_intp, dUa_intp, dUc_intp = batteqns.get_smooth_Uref_data( Ua_path, Uc_path, ffa=0.1, ffc=0.1, filter_on=1 )

xa = numpy.linspace( 0.001, 0.999, 200 ) #raw_Ua[:,0] #
Ua = Ua_intp(xa)

xc = numpy.linspace( 0.001, 0.999, 200 ) # raw_Uc[:,0] #
Uc = Uc_intp(xc)

numpy.savetxt( Ua_path[:-4]+'_refx.csv', numpy.array( [xa, Ua] ).T, delimiter=',' )
numpy.savetxt( Uc_path[:-4]+'_refx.csv', numpy.array( [xc, Uc] ).T, delimiter=',' )

#plt.figure(1)
#plt.plot( raw_Ua[:,0], raw_Ua[:,1], label='raw_Ua' )
#plt.plot( xa, Ua, label='filt_Ua' )
#plt.legend()

#plt.figure(2)
#plt.plot( raw_Uc[:,0], raw_Uc[:,1], label='raw_Uc' )
#plt.plot( xc, Uc, label='filt_Uc' )
#plt.legend()

#plt.show()
