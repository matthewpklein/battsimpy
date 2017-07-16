import pickle

from matplotlib import pyplot as plt
plt.style.use('classic')

import matplotlib as mpl

fs = 12.
fw = 'bold'
mpl.rc('lines', linewidth=2., color='k')
mpl.rc('font', size=fs, weight=fw, family='Arial')
mpl.rc('legend', fontsize='small')


import numpy

def grad( x, u ) :
    return numpy.gradient(u) / numpy.gradient(x)

date = '20160519'

base = '/home/mk-sim-linux/Battery_TempGrad/Python/batt_simulation/battsimpy/' 
base_dir = '/home/mk-sim-linux/Battery_TempGrad/JournalPaper2/Paper2/ocv_unif35/'

fig_dir = '/home/mk-sim-linux/Battery_TempGrad/JournalPaper3/modeling_paper_p3/figs/'

#base_dir = '/home/m_klein/tgs_data/ocv_unif35/'
#base_dir = '/Volumes/Data/Paper2/ocv_dat/'
#bsp_path = '/Users/mk/Desktop/battsim/battsimpy/'

nmc_rest_523    = numpy.loadtxt( base+'data/Model_nmc/Model_Pars/solid/thermodynamics/2012Yang_523NMC_dchg_restOCV.csv', delimiter=',' )
nmc_cby25_111   = numpy.loadtxt( base+'data/Model_nmc/Model_Pars/solid/thermodynamics/2012Wu_NMC111_Cby25_dchg.csv'    , delimiter=',' )
nmc_YangWu_mix  = numpy.loadtxt( base+'data/Model_nmc/Model_Pars/solid/thermodynamics/YangWuMix_NMC_20170607.csv'      , delimiter=',' )
lfp_prada_dchg  = numpy.loadtxt( base+'data/Model_v1/Model_Pars/solid/thermodynamics/2012Prada_LFP_U_dchg.csv'        , delimiter=',' )
graph_hess_dchg = numpy.loadtxt( base+'data/Model_nmc/Model_Pars/solid/thermodynamics/Ua_cell4Fit_NMC_2012Yang_refx.csv'   , delimiter=',' ) #graphite_Hess_discharge_x.csv


#xin, Uin = 1.-lfp_prada_dchg[:,0], lfp_prada_dchg[:,1]
#xin, Uin = 1.-nmc_rest_523[:,0], nmc_rest_523[:,1]
xin, Uin = 1.-nmc_YangWu_mix[:,0], nmc_YangWu_mix[:,1]
#xin, Uin = 1.-nmc_cby25_111[:,0], nmc_cby25_111[:,1]

xin2, Uin2 = graph_hess_dchg[:,0], graph_hess_dchg[:,1]#-0.025

pfiles2 = [ base_dir+'slowOCVdat_cell4_slow_ocv_'+date+'.p', ]

# Load the cell ocv c/60 data
d = pickle.load( open( pfiles2[0], 'rb' ) )
max_cap = numpy.amax( d['interp']['cap'] )

x_cell, U_cell = 1-numpy.array(d['interp']['cap'])/max_cap*1., d['interp']['dchg']['volt']

# NMC 532 scale - NMC cyl cells (cell 4)
#scale_x = 1.8#1.5 # 1.55
#shift_x = -.01#-.06 #-.12
scale_x = 1.42 # 1.55
shift_x = -.03 #-.12


#scale_x1 = 1.9
#shift_x1 = -.03

## LFP Prada - (cell 2)
#scale_x =  1.25
#shift_x = 1.05-scale_x

# Graphite - scale NMC cyl cells (cell 4)
scale_x2 = 1/.8 #1./0.83 #
shift_x2 = -.06 #-.035
#scale_x2 = 1/.74
#shift_x2 = -.04

figres = 300
figname = base_dir+'ocv-plots_'+date+'.pdf'

sty = [ '-', '--' ]
fsz = (190./25.4,120./25.4)
f1, axes = plt.subplots(1,2,figsize=fsz)

a1,a2 = axes
# Plot the full cell ocv
a1.plot( x_cell, U_cell, '-b', label='Cell C/60 Data'   )

# Plot the cathode curve for the shifted soc operating window
a1.plot( xin*scale_x+shift_x, Uin, '-g', label='Cathode'   )

# Plot the anode curve for the shifted soc operating window
#a1t = a1.twinx()
a1.plot( xin2*scale_x2+shift_x2, Uin2, '-k', label='Anode'   )

# Compute the cathode ocv for the full cell soc operating window
if xin[1] < xin[0] :
    Uc = numpy.interp( x_cell, numpy.flipud(xin*scale_x+shift_x), numpy.flipud(Uin) )
else :
    Uc = numpy.interp( x_cell, xin*scale_x+shift_x, Uin )

Ua = numpy.interp( x_cell, xin2*scale_x2+shift_x2, Uin2 )

# Plot the estimated full cell ocv curve for the aligned anode and cathode equilibrium curves
#a1.plot( x_cell, Uc-U_cell, ':k', label='U$_{anode}$ fit'   )
#a1t.set_ylim([0.,2.])
a1.plot( x_cell, Uc-Ua, ':k', label='U$_{cell}$ fit'   )

# Calculate the alignment stoichs for anode and cathode
Ua_out = Uc - U_cell
xa_out = (x_cell-shift_x2)/scale_x2
#numpy.savetxt( base+'data/Model_v1/Model_Pars/solid/thermodynamics/Ua_lfp_2012Prada.csv', numpy.array([xa_out, Ua_out]).T, delimiter=',' )
#numpy.savetxt( base+'data/Model_v1/Model_Pars/solid/thermodynamics/Ua_nmc_2012Yang.csv', numpy.array([xa_out, Ua_out]).T, delimiter=',' )

yin = 1.-xin

xc_lo = 1. - (-shift_x/scale_x)
xc_hi = 1. - (1.-shift_x)/scale_x

xa_lo = (-shift_x2/scale_x2)
xa_hi = (1.-shift_x2)/scale_x2

# Print out the stoich limits for the anode and cathode
print 'xc_lo, xc_hi:',xc_lo, xc_hi
print 'xa_lo, xa_hi:',xa_lo, xa_hi

a1.set_xlabel( 'State of Charge', fontsize=fs, fontweight=fw )
a1.set_ylabel( 'Voltage vs. Li [V]', fontsize=fs, fontweight=fw )
a1.set_title(  'Full and Half Cell OCV', fontsize=fs, fontweight=fw )
a1.legend(loc='best')
a1.set_axisbelow(True)
a1.grid(color='gray')

a2.plot( x_cell, grad(x_cell, U_cell), label=r'$\frac{\partial U_{cell}}{\partial SOC}$' )
a2.plot( x_cell, -grad(x_cell, Ua), label=r'$\frac{\partial U_{anode}}{\partial SOC}$' )
a2.set_xlabel( 'State of Charge', fontsize=fs, fontweight=fw )
a2.set_ylabel( '$\partial U / \partial SOC$', fontsize=fs, fontweight=fw )
a2.set_title(  'OCV Gradients for Anode Alignment', fontsize=fs, fontweight=fw )
a2.legend(loc='best')
a2.set_axisbelow(True)
a2.grid(color='gray')
a2.set_ylim([-0.1,1.5])

#plt.suptitle('LFP/C$_6$ Half Cell OCV Alignment', fontsize=fs, fontweight=fw)
plt.suptitle('NMC/C$_6$ Half Cell OCV Alignment', fontsize=fs, fontweight=fw)

plt.tight_layout(rect=[0,0.03,1,0.97])

plt.show()
#f1.savefig( fig_dir+'ocv_alignment_cell2_lfp.pdf', dpi=figres)
#f1.savefig( fig_dir+'ocv_alignment_cell4_nmc.pdf', dpi=figres)

