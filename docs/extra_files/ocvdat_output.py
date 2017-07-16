import pickle
import matplotlib as mpl
fs = 16.
fw = 'bold'
mpl.rc('lines', linewidth=2., color='k')
mpl.rc('font', family='serif', size=fs, weight=fw)
mpl.rc('legend', fontsize='small')
from matplotlib import pyplot as plt
import numpy

date = '20160519'
base_dir = '/home/mk-sim-linux/Battery_TempGrad/JournalPaper2/Paper2/ocv_unif35/'
#base_dir = '/home/m_klein/tgs_data/ocv_unif35/'
#base_dir = '/Volumes/Data/Paper2/ocv_dat/'
#bsp_path = '/Users/mk/Desktop/battsim/battsimpy/'

#pfiles1 = [base_dir+'slowOCVdat_cell1_slow_ocv_'+date+'.p', base_dir+'slowOCVdat_cell2_slow_ocv_'+date+'.p' ]
#pfiles2 = [base_dir+'slowOCVdat_cell3_slow_ocv_'+date+'.p', base_dir+'slowOCVdat_cell4_slow_ocv_'+date+'.p' ]
#pfiles1 = [ base_dir+'slowOCVdat_cell2_slow_ocv_'+date+'.p', ]
pfiles2 = [ base_dir+'slowOCVdat_cell4_slow_ocv_'+date+'.p', ]

figres = 300
figname = base_dir+'ocv-plots_'+date+'.pdf'

sty = [ '-', '--' ]
fsz = (10, 5)
f1 = plt.figure(1) #,figsize=fsz, dpi=figres )

d = pickle.load( open( pfiles2[0], 'rb' ) )

max_cap = numpy.amax( d['interp']['cap'] )

i=0

plt.plot( d['interp']['cap'], d['interp']['dchg']['volt'], sty[i]+'r', label='Cell-'+str(i+1)+'$_{dchg}$'   )

numpy.savetxt( './ocv_dat_cell4.csv', numpy.array([d['interp']['cap'],d['interp']['dchg']['volt']]).T, delimiter=',' )

plt.show()
