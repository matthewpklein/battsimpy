import confreader

path = '/Users/mk/Desktop/battsim/battsimpy/data/Model_v1/Model_Pars/des_prop.txt'
#path = '/Users/mk/Desktop/battsim/battsimpy/testConf.conf'

testdat = confreader.reader( path )


print testdat.conf_data

