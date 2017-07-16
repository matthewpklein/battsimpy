import model

confdir    = '/home/mk-sim-linux/Battery_TempGrad/Python/batt_simulation/battsimpy/configs/'

#confdir    = '/Users/mk/Desktop/battsim/battsimpy/configs/'

#configFile = confdir+'DCRconf_macTesting.conf'

#full_1d = model.MODEL(configFile)
#full_1d.buildmodel()
#full_1d.simulate()
#full_1d.plotresults()

#configFile = confdir+'DCRconf_macTesting_spmdist.conf'
configFile = confdir+'DCRconf_testConf_spmdist.conf'
#configFile = confdir+'CCconf_testConf_spmdist.conf'
#configFile = confdir+'CCconf_macTesting_spmdist.conf'
#configFile = confdir+'CCconf_macTesting_spm.conf'

#configFile = confdir+'DCRconf_macTesting_spm.conf'
#configFile = confdir+'CCconf_testConf_spm.conf'

#configFile = confdir+'HPPCconf_testConf_spmdist.conf'

spm = model.MODEL(configFile)

spm.buildmodel()

spm.loadresults()

spm.plotresults()
