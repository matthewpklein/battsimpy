import confreader
import params
import full_1d
import batteqns

import math
import numpy
from matplotlib import pyplot as plt

def test_init( basepath, confname ) :

    # Config data load
    path = basepath+'battsimpy/'+confname
    cfr = confreader.reader( path )
    RunInput = cfr.conf_data

    f1d = full_1d.FULL_1D( RunInput )

    for key in f1d.varinf :
        print '#########################################'
        print '~~~  ', key, '  ~~~'
        print 
        for subkey in f1d.varinf[key] :
            print '\t~~~  ', subkey, '  ~~~'
            print '\t',f1d.varinf[key][subkey]
            print 


def test_phismats( basepath, confname ) :

    # Config data load
    path = basepath+'battsimpy/'+confname
    cfr = confreader.reader( path )
    RunInput = cfr.conf_data

    # Initialize the model
    f1d = full_1d.FULL_1D( RunInput )

    # Execute the phi_s_mats function
    p = f1d.pars

    x = [p.A_psn, p.B_psn, p.A_psp, p.B_psp, p.B2_psn, p.B2_psp]
    names = ['An', 'Bn', 'Ap', 'Bp', 'B2n', 'B2p']
    for i,mat in enumerate(x) :
        print '#########################################'
        print '~~~  ', names[i], '  ~~~'
        print mat
        print 


def test_iemats( basepath, confname ) :

    # Config data load
    path = basepath+'battsimpy/'+confname
    cfr = confreader.reader( path )
    RunInput = cfr.conf_data

    # Initialize the model
    f1d = full_1d.FULL_1D( RunInput )

    # Execute the phi_s_mats function
    p = f1d.pars

    x = [p.A_ien, p.B_ien, p.A_iep, p.B_iep, p.G_ien, p.G_iep]
    names = ['An', 'Bn', 'Ap', 'Bp', 'Gn', 'Gp']
    for i,mat in enumerate(x) :
        print '#########################################'
        print '~~~  ', names[i], '  ~~~'
        print mat
        print 


def test_cspre( basepath, confname ) :

    # Config data load
    path = basepath+'battsimpy/'+confname
    cfr = confreader.reader( path )
    RunInput = cfr.conf_data

    # Initialize the model
    f1d = full_1d.FULL_1D( RunInput )

    # Execute the phi_s_mats function
    p = f1d.pars

    x = [p.A_csn_pre, p.B_csn_pre, p.A_csp_pre, p.B_csp_pre]
    names = ['An', 'Bn', 'Ap', 'Bp']
    for i,mat in enumerate(x) :
        print '#########################################'
        print '~~~  ', names[i], '  ~~~'
        print mat
        print 


def test_jacpre( basepath, confname ) :

    # Config data load
    path = basepath+'battsimpy/'+confname
    cfr = confreader.reader( path )
    RunInput = cfr.conf_data

    # Initialize the model
    f1d = full_1d.FULL_1D( RunInput )

    # Execute the jac_pre function
    f1d.jac_pre()
    p = f1d.pars

    x = [p.f_x, p.f_z, p.g_x, p.g_z]
    names = ['fx', 'fz', 'gx', 'gz']
    for i,mat in enumerate(x) :
        print '#########################################'
        print '~~~  ', names[i], '  ~~~'
        print mat
        print 


def test_cce( basepath, confname ) :

    # Config data load
    path = basepath+'battsimpy/'+confname
    cfr = confreader.reader( path )
    RunInput = cfr.conf_data

    # Initialize the model
    f1d = full_1d.FULL_1D( RunInput )
    p = f1d.pars
    
    # Execute the cce function
    c_e = numpy.ones( p.Nce )*1220.
    f1d.cce( c_e )
    print p.C_ce


def test_cemats( basepath, confname ) :

    # Config data load
    path = basepath+'battsimpy/'+confname
    cfr = confreader.reader( path )
    RunInput = cfr.conf_data

    # Initialize the model
    f1d = full_1d.FULL_1D( RunInput )
    p = f1d.pars
    
    # Execute the cce function
    cer=1220.
    c_e   = numpy.ones( p.Nce )*cer
    cex   = numpy.ones( p.Nce+4 )*cer
    cebcs = numpy.ones( 4 )*cer
    T = 298.15
    
    f1d.cce(c_e)
    
    A, B, F = f1d.c_e_mats( c_e, cebcs, cex, T )

    x = [A, B, F]
    names = ['A', 'B', 'F']
    for i,mat in enumerate(x) :
        print '#########################################'
        print '~~~  ', names[i], '  ~~~'
        print mat
        print 


def test_phiemats( basepath, confname ) :

    # Config data load
    path = basepath+'battsimpy/'+confname
    cfr = confreader.reader( path )
    RunInput = cfr.conf_data

    # Initialize the model
    f1d = full_1d.FULL_1D( RunInput )
    p = f1d.pars
    
    # Execute the cce function
    cer=1220.
    c_e   = numpy.ones( p.Nce )*cer
    cex   = numpy.ones( p.Nce+4 )*cer
    cebcs = numpy.ones( 4 )*cer
    T = 298.15
    
    f1d.cce(c_e)
    
    F1, F2, F3, C_phie, C_phie_cex, phiemats = f1d.phi_e_mats( c_e, cebcs, cex, T )

    x = [F1, F2, F3, C_phie, C_phie_cex]
    names = ['F1', 'F2', 'F3', 'Cphie', 'Cphiex']
    for i,mat in enumerate(x) :
        print '#########################################'
        print '~~~  ', names[i], '  ~~~'
        print mat
        print 


def test_soldiff( basepath, confname ) :

    # Config data load
    path = basepath+'battsimpy/'+confname
    cfr = confreader.reader( path )
    RunInput = cfr.conf_data

    # Initialize the model
    f1d = full_1d.FULL_1D( RunInput )
    p = f1d.pars
    
    # Setup the solid_diffusion function inputs
    elec = 'cathode'
    if elec == 'cathode':
        x, r = p.Np, p.SolidOrder_p
        csmax = p.c_s_p_max
        asf = p.a_s_p*p.L_p
        csdat, Dsdat = p.csdat_p, p.Dsdat_p
        theta_0 = 0.3596914899
    elif elec == 'anode' :
        x, r = p.Nn, p.SolidOrder_n
        csmax = p.c_s_n_max
        asf = p.a_s_n*p.L_n
        csdat, Dsdat = p.csdat_n, p.Dsdat_n
        theta_0 = 0.83058516056702
    
    cs_sz = x*r
    csr   = theta_0*csmax
    c_s   = numpy.ones( cs_sz, dtype='d' )*csr
    j_rxn = (1./(p.Area*(asf)))/p.Faraday * numpy.zeros( x, dtype='d' )
    msz = ( x, r )
    T = 298.15

    c_s_dot, c_ss, c_avg, As, Bs = f1d.solid_diffusion( c_s, j_rxn, T, csdat, Dsdat, msz )

    x = [c_s_dot[0:p.Np], c_ss, c_avg, As[:,:,0], Bs[:,0]]
    names = ['c_s_dot', 'c_ss', 'c_avg', 'As', 'Bs']
#    x = [c_ss, c_avg]
#    names = ['c_ss', 'c_avg']
    for i,mat in enumerate(x) :
        print '#########################################'
        print '~~~  ', names[i], '  ~~~'
        print mat
        print 


def test_elytediff( basepath, confname ) :

    # Config data load
    path = basepath+'battsimpy/'+confname
    cfr = confreader.reader( path )
    RunInput = cfr.conf_data

    # Initialize the model
    f1d = full_1d.FULL_1D( RunInput )
    p = f1d.pars
    
    # Execute the cce function
    cer=1220.
#    c_e = numpy.ones( p.Nce, dtype='d' )*cer
    c_e = numpy.linspace( .9, 1.1, num=p.Nce, dtype='d' )*cer
    i_e = numpy.zeros_like( c_e, dtype='d' )

    T = 298.15

    Cur = 0.0

    f1d.cce(c_e)

    c_e_dot, cemats, ce_bcs, c_ex = f1d.electrolyte_diffusion( c_e, i_e, Cur, T )

    x = [c_e_dot, cemats, ce_bcs, c_ex]
    names = ['c_e_dot', 'cemats', 'ce_bcs', 'c_ex']
#    x = [c_ss, c_avg]
#    names = ['c_ss', 'c_avg']
    for i,mat in enumerate(x) :
        print '#########################################'
        print '~~~  ', names[i], '  ~~~'
        print mat
        print 


def test_getstates( basepath, confname ) :

    # Config data load
    path = basepath+'battsimpy/'+confname
    cfr = confreader.reader( path )
    RunInput = cfr.conf_data

    # Initialize the model
    f1d = full_1d.FULL_1D( RunInput )
    p = f1d.pars

    # Create a dummy x and z
    csn0, csp0, ce0, T0 = 0.5*p.c_s_n_max, 0.5*p.c_s_p_max, 1200., 298.15
    x = numpy.concatenate( (csn0*numpy.ones(p.Nn*p.SolidOrder_n), csp0*numpy.ones(p.Np*p.SolidOrder_p), ce0*numpy.ones(p.Nce), (T0,) ) )

    phisn0, phisp0, ien0, iep0, phie0, jn0, jp0 = 0.05, 3.8, 0.1, 0.2, 0.3, 0.4, 0.5
    z = numpy.concatenate( (phisn0*numpy.ones(p.Nn), phisp0*numpy.ones(p.Np), ien0*numpy.ones(p.Nn), iep0*numpy.ones(p.Np), phie0*numpy.ones(p.Nce), jn0*numpy.ones(p.Nn), jp0*numpy.ones(p.Np) ) )

    # Run the function
    states = f1d.get_states( x, z )

#    x = [c_ss, c_avg]
#    names = ['c_ss', 'c_avg']
    for k in states :
        print '#########################################'
        print '~~~  ', k, '  ~~~'
        print states[k]
        print 

def test_getEdgeEtas( basepath, confname ) :

    # Config data load
    path = basepath+'battsimpy/'+confname
    cfr = confreader.reader( path )
    RunInput = cfr.conf_data

    # Initialize the model
    f1d = full_1d.FULL_1D( RunInput )
    p = f1d.pars
    
    # 
    T0 = 298.15
#    io = [10., 1.]
    aFRT_n, aFRT_p = 0.5*p.Faraday / ( p.R*T0 ), 0.5*p.Faraday / ( p.R*T0 )
    Rf = 0.0
    j_rxn0, j_rxnN = (1./(p.Area*(p.a_s_p*p.L_p)))/p.Faraday         ,  (1./(p.Area*(p.a_s_p*p.L_p)))/p.Faraday*1.1
    eta1  , etaNm1 = (1./aFRT_n) * math.asinh( p.Faraday/(2*io[0])*j_rxn0 ),  (1./aFRT_n) * math.asinh( p.Faraday/(2*io[-1])*j_rxnN )

    eta0, etaN = f1d.getEdgeEtas( io, aFRT_n, aFRT_p, Rf, j_rxn0, j_rxnN, eta1, etaNm1 )

    x = [eta0, eta1, etaNm1, etaN]
    names = ['eta0', 'eta1', 'etaNm1', 'etaN']
    for i,mat in enumerate(x) :
        print '#########################################'
        print '~~~  ', names[i], '  ~~~'
        print mat
        print 


def test_daesystem( basepath, confname ) :

    # Config data load
    path = basepath+'battsimpy/'+confname
    cfr = confreader.reader( path )
    RunInput = cfr.conf_data

    # Initialize the model
    f1d = full_1d.FULL_1D( RunInput )
    p = f1d.pars
    
    T = 298.15

    Cur = 0.0
    
    # Create a dummy x and z
    csn0, csp0, ce0, T0 = p.theta_n0*p.c_s_n_max, p.theta_p0*p.c_s_p_max, 1220., T
    c_e = ce0*numpy.ones(p.Nce, dtype='d')
#    csn = csn0*numpy.ones(p.Nn*p.SolidOrder_n, dtype='d')
#    csp = csp0*numpy.ones(p.Np*p.SolidOrder_p, dtype='d')
    csn = numpy.linspace( (1.0*p.theta_n0)*p.c_s_n_max, (1.0*p.theta_n0)*p.c_s_n_max, num=p.Nn*p.SolidOrder_n, dtype='d')
    csp = numpy.linspace( (1.0*p.theta_p0)*p.c_s_p_max, (1.0*p.theta_p0)*p.c_s_p_max, num=p.Np*p.SolidOrder_p, dtype='d')
    x = numpy.concatenate( (csn, csp, c_e, (T0,) ) )

    print '    x_p0','       x_n0'
    print p.theta_p0, p.theta_n0

    print csn[0:p.Nn]/p.c_s_n_max

    Un0 = batteqns.refPotential( p.Udat_n, csn[0:p.Nn]/p.c_s_n_max, T )
    Up0 = batteqns.refPotential( p.Udat_p, csp[0:p.Np]/p.c_s_p_max, T )
    
    print '     Up0','       Un0'
    print Up0, Un0

    phisn0, phisp0, ien0, iep0, phie0, jn0, jp0 = Un0, Up0, 0.0, 0.0, 0.0, 0.0, 0.0
    z = numpy.concatenate( (phisn0, phisp0, 
                            ien0*numpy.ones(p.Nn, dtype='d'), 
                            iep0*numpy.ones(p.Np, dtype='d'), 
                            phie0*numpy.ones(p.Nce, dtype='d'), 
                            jn0*numpy.ones(p.Nn, dtype='d'), 
                            jp0*numpy.ones(p.Np, dtype='d') ) )

    f1d.cce(c_e)

    f, g = f1d.dae_system( x, z, Cur, get_derivs=1 )

    x = [f, g]
    names = ['f', 'g']
    for i,mat in enumerate(x) :
        print '#########################################'
        print '~~~  ', names[i], '  ~~~'
        print mat
        print 


def test_jacsystem( basepath, confname ) :

    # Config data load
    path = basepath+'battsimpy/'+confname
    cfr = confreader.reader( path )
    RunInput = cfr.conf_data

    # Initialize the model
    f1d = full_1d.FULL_1D( RunInput )
    p = f1d.pars
    
    T = 298.15

    Cur = 0.0
    
    # Create a dummy x and z
    csn0, csp0, ce0, T0 = p.theta_n0*p.c_s_n_max, p.theta_p0*p.c_s_p_max, 1220., T
    c_e = ce0*numpy.ones(p.Nce, dtype='d')
#    csn = csn0*numpy.ones(p.Nn*p.SolidOrder_n, dtype='d')
#    csp = csp0*numpy.ones(p.Np*p.SolidOrder_p, dtype='d')
    csn = numpy.linspace( (1.0*p.theta_n0)*p.c_s_n_max, (1.0*p.theta_n0)*p.c_s_n_max, num=p.Nn*p.SolidOrder_n, dtype='d')
    csp = numpy.linspace( (1.0*p.theta_p0)*p.c_s_p_max, (1.0*p.theta_p0)*p.c_s_p_max, num=p.Np*p.SolidOrder_p, dtype='d')
    x = numpy.concatenate( (csn, csp, c_e, (T0,) ) )

    print '    x_p0','       x_n0'
    print p.theta_p0, p.theta_n0

    print csn[0:p.Nn]/p.c_s_n_max

    Un0 = batteqns.refPotential( p.Udat_n, csn[0:p.Nn]/p.c_s_n_max, T )
    Up0 = batteqns.refPotential( p.Udat_p, csp[0:p.Np]/p.c_s_p_max, T )
    
    print '     Up0','       Un0'
    print Up0, Un0

    phisn0, phisp0, ien0, iep0, phie0, jn0, jp0 = Un0, Up0, 0.0, 0.0, 0.0, 0.0, 0.0
    z = numpy.concatenate( (phisn0, phisp0, 
                            ien0*numpy.ones(p.Nn, dtype='d'), 
                            iep0*numpy.ones(p.Np, dtype='d'), 
                            phie0*numpy.ones(p.Nce, dtype='d'), 
                            jn0*numpy.ones(p.Nn, dtype='d'), 
                            jp0*numpy.ones(p.Np, dtype='d') ) )

    f1d.cce(c_e)

    f, g, solid_mats, phie_mats, ce_mats, var_flag = f1d.dae_system( x, z, Cur, get_mats=1 )


    fx, fz, gx, gz = f1d.jac_system( x, z, Cur, solid_mats, phie_mats, ce_mats )

    x = [fx, fz, gx, gz]
    names = ['fx', 'fz', 'gx', 'gz']
    for i,mat in enumerate(x) :
        print '#########################################'
        print '~~~  ', names[i], '  ~~~'
        print 'Shape:', mat.shape
        print 'Min:', numpy.amin(mat), 'Max:', numpy.amax(mat)
        print mat
        print 
        if names[i] == 'fz' :
            print mat[779,308]


def test_cnsolver( basepath, confname ) :

    # Config data load
    path = basepath+'battsimpy/'+confname
    cfr = confreader.reader( path )
    RunInput = cfr.conf_data

    # Initialize the model
    f1d = full_1d.FULL_1D( RunInput )
    p = f1d.pars
    
    T = 298.15

    Cur = 0.0
    
    # Create a dummy x and z
    csn0, csp0, ce0, T0 = p.theta_n0*p.c_s_n_max, p.theta_p0*p.c_s_p_max, 1220., T
    c_e = ce0*numpy.ones(p.Nce, dtype='d')
#    csn = csn0*numpy.ones(p.Nn*p.SolidOrder_n, dtype='d')
#    csp = csp0*numpy.ones(p.Np*p.SolidOrder_p, dtype='d')
    csn = numpy.linspace( (1.0*p.theta_n0)*p.c_s_n_max, (1.0*p.theta_n0)*p.c_s_n_max, num=p.Nn*p.SolidOrder_n, dtype='d')
    csp = numpy.linspace( (1.0*p.theta_p0)*p.c_s_p_max, (1.0*p.theta_p0)*p.c_s_p_max, num=p.Np*p.SolidOrder_p, dtype='d')
    x = numpy.concatenate( (csn, csp, c_e, (T0,) ) )

#    print '    x_p0','       x_n0'
#    print p.theta_p0, p.theta_n0

#    print csn[0:p.Nn]/p.c_s_n_max

    Un0 = batteqns.refPotential( p.Udat_n, csn[0:p.Nn]/p.c_s_n_max, T )
    Up0 = batteqns.refPotential( p.Udat_p, csp[0:p.Np]/p.c_s_p_max, T )
#    
#    print '     Up0','       Un0'
#    print Up0, Un0

    phisn0, phisp0, ien0, iep0, phie0, jn0, jp0 = Un0, Up0, 0.0, 0.0, 0.0, 0.0, 0.0
    z = numpy.concatenate( (phisn0, phisp0, 
                            ien0*numpy.ones(p.Nn, dtype='d'), 
                            iep0*numpy.ones(p.Np, dtype='d'), 
                            phie0*numpy.ones(p.Nce, dtype='d'), 
                            jn0*numpy.ones(p.Nn, dtype='d'), 
                            jp0*numpy.ones(p.Np, dtype='d') ) )

    f1d.cce(c_e)

    Cur_vec = numpy.array( [ Cur, Cur, Cur ], dtype='d' )

    x_nxtf, z_nxtf, newtonStats = f1d.cn_solver( x, z, Cur_vec ) 


    states = f1d.get_states( x_nxtf, z_nxtf )

    plt.figure(1)
    plt.plot(states['phi_e'])
    plt.ylim([-1,1])
#    plt.plot(states['c_e'])
#    plt.ylim([1219,1221])
    plt.show()

    x = [x_nxtf, z_nxtf, newtonStats]
    names = ['x_nxtf', 'z_nxtf', 'newtonStats']
    for i,mat in enumerate(x) :
        print '#########################################'
        print '~~~  ', names[i], '  ~~~'
        print mat
        print 


confname = 'testConf.conf'
basepath = '/home/mk-sim-linux/Battery_TempGrad/Python/batt_simulation/'

#confname = 'testConf_mac.conf'
#basepath = '/Users/mk/Desktop/battsim/'

#test_init( basepath, confname )

#test_cspre( basepath, confname )

#test_phismats( basepath, confname )

#test_iemats( basepath, confname )

#test_jacpre( basepath, confname )

#test_cce( basepath, confname )

#test_cemats( basepath, confname )

#test_phiemats( basepath, confname )

#test_soldiff( basepath, confname )

#test_elytediff( basepath, confname )

#test_getstates( basepath, confname )

#test_getEdgeEtas( basepath, confname ) 

#test_daesystem( basepath, confname ) 

#test_jacsystem( basepath, confname ) 

test_cnsolver( basepath, confname ) 
