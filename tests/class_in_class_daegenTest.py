import dae_genPart_T as dgt

d = dgt.wraptest()
d.simulate()
#
#import confreader
#import full_1d_fvm_ida
##import params
#
#confdir    = '/home/mk-sim-linux/Battery_TempGrad/Python/batt_simulation/battsimpy/configs/'
#
#configFile = confdir+'model_fvmP2D.conf'
#simFile    = confdir+'sim_DCR.conf'
#
#mcd = confreader.reader( configFile )
#scd = confreader.reader( simFile )
#
#conf_data = mcd.conf_data.copy()
#conf_data.update(scd.conf_data)
#
##model = full_1d_fvm_ida.FULL_1D( conf_data )
##model.model_setup()
##
##num_steps, num_cycs = 1, 1
##
##model.build_results_dict( num_steps, num_cycs )
#
#simulator = full_1d_fvm_ida.simulator( conf_data )
#
#NT = 80
#tfinal = 10.0
#inp = 10.0
#init_rest_on = 1
#last_run_name = ''
#run_name = 'step0_repeat0'
#
#
#results = simulator.simulate( tfinal, NT, inp, init_rest_on, run_name, last_run_name )
#

