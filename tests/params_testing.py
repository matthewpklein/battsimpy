import params
import confreader

from matplotlib import pyplot as plt

def matl_prop_testing( basepath ) :

    path_matl = basepath+'battsimpy/data/Model_v1/Model_Pars/matl_prop.txt'
    path_des  = basepath+'battsimpy/data/Model_v1/Model_Pars/des_prop.txt'
    
    a = params.params()

    a.get_matl_properties( path_matl )
    
    print '==============================='
    print a.matl_prop
    print '==============================='

def des_prop_testing( basepath ) :

    path_matl = basepath+'battsimpy/data/Model_v1/Model_Pars/matl_prop.txt'
    path_des  = basepath+'battsimpy/data/Model_v1/Model_Pars/des_prop.txt'
    
    a = params.params()
    a.get_matl_properties( path_matl ) # matl_prop must be run first


    a.get_des_properties( path_des )
    
    print '==============================='
    print a.des_prop
    print '==============================='


def test_refPots( basepath, confname ) :

    # Config data load
    path = basepath+'battsimpy/'+confname
    cfr = confreader.reader( path )
    RunInput = cfr.conf_data

    Pdat = { 'RunInput':RunInput }
    V_init = 4.1
    
    p = params.params()
    p.buildpars( V_init, Pdat )

    plt.figure(1)
    plt.plot( p.Udat_p['x'], p.Udat_p['U'], label='cathode' )
    plt.plot( p.Udat_n['x'], p.Udat_n['U'], label='anode' )
    plt.legend()
    plt.show()


def test_thermFactors( basepath, confname ) :

    # Config data load
    path = basepath+'battsimpy/'+confname
    cfr = confreader.reader( path )
    RunInput = cfr.conf_data

    Pdat = { 'RunInput':RunInput }
    V_init = 4.1
    
    p = params.params()
    p.buildpars( V_init, Pdat )

    plt.figure(1)
    plt.plot( p.Udat_p['x'], p.Udat_p['activity'], label='cathode' )
    plt.plot( p.Udat_n['x'], p.Udat_n['activity'], label='anode' )
    plt.legend()
    plt.show()


def test_fullocp( basepath, confname ) :

    # Config data load
    path = basepath+'battsimpy/'+confname
    cfr = confreader.reader( path )
    RunInput = cfr.conf_data

    Pdat = { 'RunInput':RunInput }
    V_init = 4.1
    
    p = params.params()
    p.buildpars( V_init, Pdat )

    print 'xn0:', p.theta_n0
    print 'xp0:', p.theta_p0

    plt.figure(1)
    plt.plot( p.xs, p.Up, label='U_p' )
    plt.plot( p.xs, p.Ua, label='U_n' )
    plt.plot( p.xs, p.OCP, label='OCP' )
    plt.legend()
    plt.show()

def test_precsAmats( basepath, confname ) :

    # Config data load
    path = basepath+'battsimpy/'+confname
    cfr = confreader.reader( path )
    RunInput = cfr.conf_data

    Pdat = { 'RunInput':RunInput }
    V_init = 4.1
    
    p = params.params()
    p.buildpars( V_init, Pdat )

    x = [p.A1n, p.A2n, p.A1p, p.A2p]
    names = ['A1n', 'A2n', 'A1p', 'A2p']
    for i,mat in enumerate(x) :
        print '#########################################'
        print '~~~  ', names[i], '  ~~~'
        print mat
        print 


def params_testing( basepath, confname ) :
    """
    Checks to see that the buildpars method in params will run. Does not do anything
    to ensure the correctness of the data
    """
    # Config data load
    path = basepath+'battsimpy/'+confname
    cfr = confreader.reader( path )
    RunInput = cfr.conf_data

    Pdat = { 'RunInput':RunInput }
    V_init = 4.1
    
    a = params.params()
    a.buildpars( V_init, Pdat )

#    b = a.De_intp_table.ev([3000.,2000.,1000.],[298.,298.,280.])


############################

confname = 'testConf.conf'
basepath = '/home/mk-sim-linux/Battery_TempGrad/Python/batt_simulation/'

#confname = 'testConf_mac.conf'
#basepath = '/Users/mk/Desktop/battsim/'

#matl_prop_testing(basepath, confname)

#des_prop_testing(basepath, confname)

#params_testing(basepath, confname)

#test_refPots( basepath, confname )

#test_thermFactors( basepath, confname )

#test_fullocp(basepath, confname)

test_precsAmats( basepath, confname )
