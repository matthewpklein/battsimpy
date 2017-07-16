import numpy
from matplotlib import pyplot as plt

def load_csv_to_dict( fname ) :

    with open( fname, 'r' ) as fil:

        for ir, row in enumerate(fil) :

            if ir == 0 :
                header = [ v.strip() for v in row.split(',')]

                outdict = dict( [ (lbl,[]) for lbl in header ] )

            else :
                data = [ float(v) for v in row.split(',') ]
                for v, h in zip(data,header) :
                    outdict[h].append(v)

    return outdict

def load_csv_to_mat_tintp( fname, tintp_lims, NT ) :

    with open( fname, 'r' ) as fil:

        for ir, row in enumerate(fil) :

            if ir == 0 :
                head   = [ v.strip() for v in row.split(',')]
                header = numpy.array(head[1:], dtype='d')

#                print header

                outdat = []

            else :
                data = [ float(v) for v in row.split(',') ]
                outdat.append(data)

#        print outdat[0][1:]

    od = numpy.array(outdat, dtype='d')

    tintp = numpy.linspace( tintp_lims[0], tintp_lims[1], NT )

    intp_dat = numpy.array([numpy.interp( tintp, od[:,0], od[:,i+1] ) for i in range(len(header))]).T

    outdat = numpy.insert( intp_dat, 0, tintp, axis=1 )

    return header, outdat

def load_csv_to_mat_xintp( fname, xintp_lims, NX ) :

    with open( fname, 'r' ) as fil:

        for ir, row in enumerate(fil) :

            if ir == 0 :
                head   = [ v.strip() for v in row.split(',')]
                header = numpy.array(head[1:], dtype='d')

                if max(header) < 1. :
                    header = header*1e6

                outdat = []

            else :
                data = [ float(v) for v in row.split(',') ]
                outdat.append(data)

    od = numpy.array(outdat, dtype='d')
    t = od[:,0]
    xintp = numpy.linspace( xintp_lims[0], xintp_lims[1], NX )
    intp_dat = numpy.array([numpy.interp( xintp, header, od[i,1:] ) for i in range(len(od[:,0]))]).T
    outdat = numpy.insert( intp_dat, 0, t, axis=0 )

    return header, outdat.T


##########
# Inputs #
##########
Crate_names = [ ['5A','D_5A'], ['14A','D_Cby2'], ['29A','D_1C'], ['58A','D_2C'], ['87A','D_3C'], ['116A','D_4C'] ]

irate = 2

NT=5
tintp_lims = [5.,29./float(Crate_names[irate][0][:-1])*3600.*0.95]

NX=5
#xintp_lims = [2.,68./2., 68.+25.+52./2.,68.+25.+52.-2.]
xintp_lims = [2.,68.+25.+52.-2.]

mpk_cellname='uref_simba_0'#'val00_constElyte_epsa0p9_'
test_name = 'allrates' #'Test_allRates'
cell_name = '28Ah_simba_simp' #'29Ah_valid00_ConstPar_simple'
test_num  = '__1'
##########

# Load the files
f_mpk = '../outdata/20170510/'+mpk_cellname+'_v_'+Crate_names[irate][0]+'dchg.csv'
f_bds = '../outdata/20170510/potentials_output_['+test_name+']_['+cell_name+']_['+Crate_names[irate][1]+']'+test_num+'.csv'

f_list = [ f_mpk, f_bds ]
l_list = [ 'mpk','bds' ]
d_list = [ load_csv_to_dict( f ) for f in f_list ]

pe_mpk = '../outdata/20170510/'+mpk_cellname+'_pe_'+Crate_names[irate][0]+'dchg.csv'
pe_bds = '../outdata/20170510/phi_e_all_output_['+test_name+']_['+cell_name+']_['+Crate_names[irate][1]+']'+test_num+'.csv'

ce_mpk = '../outdata/20170510/'+mpk_cellname+'_ce_'+Crate_names[irate][0]+'dchg.csv'
ce_bds = '../outdata/20170510/c_e_all_output_['+test_name+']_['+cell_name+']_['+Crate_names[irate][1]+']'+test_num+'.csv'

De_mpk = '../outdata/20170510/'+mpk_cellname+'_Demid_'+Crate_names[irate][0]+'dchg.csv'
De_bds = '../outdata/20170510/De_10e10m2s_all_output_['+test_name+']_['+cell_name+']_['+Crate_names[irate][1]+']'+test_num+'.csv'

ke_mpk = '../outdata/20170510/'+mpk_cellname+'_kemid_'+Crate_names[irate][0]+'dchg.csv'
ke_bds = '../outdata/20170510/ke_Sm_all_output_['+test_name+']_['+cell_name+']_['+Crate_names[irate][1]+']'+test_num+'.csv'

eta_mpk = '../outdata/20170510/'+mpk_cellname+'_eta_fullx_'+Crate_names[irate][0]+'dchg.csv'
eta_bds = '../outdata/20170510/eta_bv_all_output_['+test_name+']_['+cell_name+']_['+Crate_names[irate][1]+']'+test_num+'.csv'

Uss_mpk = '../outdata/20170510/'+mpk_cellname+'_Uss_fullx_'+Crate_names[irate][0]+'dchg.csv'
Uss_bds = '../outdata/20170510/U_ss_all_output_['+test_name+']_['+cell_name+']_['+Crate_names[irate][1]+']'+test_num+'.csv'

fce_list = [ ce_mpk, ce_bds ]
fpe_list = [ pe_mpk, pe_bds ]
fDe_list = [ De_mpk, De_bds ]
fke_list = [ ke_mpk, ke_bds ]
eta_list = [ eta_mpk, eta_bds ]
Uss_list = [ Uss_mpk, Uss_bds ]

ce_list_tintp = [ load_csv_to_mat_tintp( f, tintp_lims, NT ) for f in fce_list ]
pe_list_tintp = [ load_csv_to_mat_tintp( f, tintp_lims, NT ) for f in fpe_list ]
De_list_tintp = [ load_csv_to_mat_tintp( f, tintp_lims, NT ) for f in fDe_list ]
ke_list_tintp = [ load_csv_to_mat_tintp( f, tintp_lims, NT ) for f in fke_list ]
eta_list_tintp = [ load_csv_to_mat_tintp( f, tintp_lims, NT ) for f in eta_list ]
Uss_list_tintp = [ load_csv_to_mat_tintp( f, tintp_lims, NT ) for f in Uss_list ]

ce_list_xintp = [ load_csv_to_mat_xintp( f, xintp_lims, NX ) for f in fce_list ]
pe_list_xintp = [ load_csv_to_mat_xintp( f, xintp_lims, NX ) for f in fpe_list ]
eta_list_xintp = [ load_csv_to_mat_xintp( f, xintp_lims, NX ) for f in eta_list ]

# Create the plots

# Plot 1, Voltages in solid
x = 'TestTime(sec)'
y_list = ['Volts','Vn','Vp','pa_cc','pc_cc','Ua_bar','Uc_bar']

plt.figure(1)
col_y = [ 'b','k','r','g','c','m','y','orange' ]
lsd   = [ '-', '--' ]
for iy, y in enumerate(y_list) :
    idat = 0
    for d, l in zip(d_list,l_list) :
        plt.plot( d[x], d[y], lsd[idat], color=col_y[iy], label=y+'_'+l )
        idat+=1

plt.xlabel('Time [sec]')
plt.ylabel('Voltage [V]')
plt.legend(loc='best',ncol=2)


# Plot 2, Concentrations in e-lyte
x = 'TestTime(sec)'
y_list = ['ce_lefta','ce_mida','ce_midsep','ce_midc', 'ce_rightc']

plt.figure(2)
col_y = [ 'b','k','r','g','c','m','y','orange' ]
lsd   = [ '-', '--' ]
for iy, y in enumerate(y_list) :
    idat = 0
    for c, l in zip(ce_list_xintp,l_list) :
        if 'bds' in l :
            plt.plot( c[1][:,0], numpy.array(c[1][:,iy+1])*1000., lsd[idat], color=col_y[iy], label=y+'_'+l )
        else :
            plt.plot( c[1][:,0], numpy.array(c[1][:,iy+1]), lsd[idat], color=col_y[iy], label=y+'_'+l )
        idat+=1

plt.xlabel('Time [sec]')
plt.ylabel('E-lyte Concentration [mol/m^3]')
plt.legend(loc='best',ncol=3)


# Plot 3, Potential in e-lyte
x = 'TestTime(sec)'
y_list = ['pe_lefta','pe_mida','pe_midsep','pe_midc', 'pe_rightc']

plt.figure(3)
col_y = [ 'b','k','r','g','c','m','y','orange' ]
lsd   = [ '-', '--' ]
for iy, y in enumerate(y_list) :
    idat = 0
    for c, l in zip(pe_list_xintp,l_list) :
        plt.plot( c[1][:,0], numpy.array(c[1][:,iy+1]), lsd[idat], color=col_y[iy], label=y+'_'+l )
        idat+=1

plt.xlabel('Time [sec]')
plt.ylabel('E-lyte Potential [V]')
plt.legend(loc='best',ncol=3)


# Plot 8, Potential in e-lyte
x = 'TestTime(sec)'
y_list = ['pe_lefta','pe_mida','pe_midsep','pe_midc', 'pe_rightc']

plt.figure(8)
col_y = [ 'b','k','r','g','c','m','y','orange' ]
lsd   = [ '-', '--' ]
for iy, y in enumerate(y_list) :
    idat = 0
    for c, l in zip(eta_list_xintp,l_list) :
        plt.plot( c[1][:,0], numpy.array(c[1][:,iy+1]), lsd[idat], color=col_y[iy], label=y+'_'+l )
        idat+=1

plt.xlabel('Time [sec]')
plt.ylabel('eta_bv Potential [V]')
plt.legend(loc='best',ncol=3)

# Plot 4, ce profiles
plt.figure(4)
lsd   = [ '-', '--' ]
idat = 0
col_y = [ 'b','k','r','g','c','m','y','orange' ]
for c,l in zip(ce_list_tintp,l_list) :
    for iy in range(len(c[1][:,1])) :
        if 'bds' in l :
            plt.plot( c[0], c[1][iy,1:]*1e3, lsd[idat], color=col_y[iy])#, label=y+'_'+l )
        else :
            plt.plot( c[0]*1e6, c[1][iy,1:], lsd[idat], color=col_y[iy])#, label=y+'_'+l )
    idat+=1

plt.xlabel('X mesh [um]')
plt.ylabel('E-lyte Conc [mol/m^3]')

# Plot 9, etabv profiles
plt.figure(9)
lsd   = [ '-', '--' ]
idat = 0
col_y = [ 'b','k','r','g','c','m','y','orange' ]
for c,l in zip(eta_list_tintp,l_list) :
    for iy in range(len(c[1][:,1])) :
        if 'bds' in l :
            plt.plot( c[0], c[1][iy,1:], lsd[idat], color=col_y[iy])#, label=y+'_'+l )
        else :
            plt.plot( c[0]*1e6, c[1][iy,1:], lsd[idat], color=col_y[iy])#, label=y+'_'+l )
    idat+=1

plt.xlabel('X mesh [um]')
plt.ylabel('eta_bv [V]')

# Plot 10, Uss profiles
plt.figure(10)
lsd   = [ '-', '--' ]
col_y = [ 'b','k','r','g','c','m','y','orange' ]
idat = 0
for c,l in zip(Uss_list_tintp,l_list) :
    for iy in range(len(c[1][:,1])) :
        if 'bds' in l :
            plt.plot( c[0], c[1][iy,1:], lsd[idat], color=col_y[iy])#, label=y+'_'+l )
        else :
            plt.plot( c[0]*1e6, c[1][iy,1:], lsd[idat], color=col_y[iy])#, label=y+'_'+l )
    idat+=1

plt.xlabel('X mesh [um]')
plt.ylabel('Uss [V]')


plt.figure(5)
col_y = [ 'b','k','r','g','c','m','y','orange' ]
lsd   = [ '-', '--' ]
for iy, y in enumerate(y_list) :
    idat = 0
    for c, l in zip(De_list_tintp,l_list) :
        if 'bds' in l :
            plt.plot( c[1][:,0], numpy.array(c[1][:,iy+1]), lsd[idat], color=col_y[iy], label=y+'_'+l )
        else :
            plt.plot( c[1][:,0], numpy.array(c[1][:,iy+1])*1e10, lsd[idat], color=col_y[iy], label=y+'_'+l )
        idat+=1

plt.xlabel('Time [sec]')
plt.ylabel('E-lyte De [1e10m^2/s]')
plt.legend(loc='best',ncol=3)

plt.figure(6)
col_y = [ 'b','k','r','g','c','m','y','orange' ]
lsd   = [ '-', '--' ]
for iy, y in enumerate(y_list) :
    idat = 0
    for c, l in zip(ke_list_tintp,l_list) :
        if 'bds' in l :
            plt.plot( c[1][:,0], numpy.array(c[1][:,iy+1]), lsd[idat], color=col_y[iy], label=y+'_'+l )
        else :
            plt.plot( c[1][:,0], numpy.array(c[1][:,iy+1]), lsd[idat], color=col_y[iy], label=y+'_'+l )
        idat+=1

plt.xlabel('Time [sec]')
plt.ylabel('E-lyte kappa [s/m]')
plt.legend(loc='best',ncol=3)

plt.figure(7)
col_y = [ 'b','k','r','g','c','m','y','orange' ]
lsd   = [ '-', '--' ]
for iy, y in enumerate(y_list) :
    idat = 0
    for c, l in zip(eta_list_tintp,l_list) :
        if 'bds' in l :
            plt.plot( c[1][:,0], numpy.array(c[1][:,iy+1]), lsd[idat], color=col_y[iy], label=y+'_'+l )
        else :
            plt.plot( c[1][:,0], numpy.array(c[1][:,iy+1]), lsd[idat], color=col_y[iy], label=y+'_'+l )
        idat+=1

plt.xlabel('Time [sec]')
plt.ylabel('eta_bv [V]')
plt.legend(loc='best',ncol=3)

plt.show()
## Plot 2, Concentrations in e-lyte
#x = 'TestTime(sec)'
#y_list = ['ce_mida','ce_midsep','ce_midc']

#plt.figure(2)
#col_y = [ 'b','k','r','g','c','m','y','orange' ]
#lsd   = [ '-', '--' ]
#for iy, y in enumerate(y_list) :
#    idat = 0
#    for d, l in zip(d_list,l_list) :
#        if 'bds' in l :
#            plt.plot( d[x], numpy.array(d[y])*1000., lsd[idat], color=col_y[iy], label=y+'_'+l )
#        else :
#            plt.plot( d[x], d[y], lsd[idat], color=col_y[iy], label=y+'_'+l )
#        idat+=1

#plt.xlabel('Time [sec]')
#plt.ylabel('E-lyte Concentration [mol/m^3]')
#plt.legend(loc='best',ncol=3)


## Plot 3, Potential in e-lyte
#x = 'TestTime(sec)'
#y_list = ['pe_mida','pe_midsep','pe_midc']

#plt.figure(3)
#col_y = [ 'b','k','r','g','c','m','y','orange' ]
#lsd   = [ '-', '--' ]
#for iy, y in enumerate(y_list) :
#    idat = 0
#    for d, l in zip(d_list,l_list) :
#        if 'bds' in l :
#            plt.plot( d[x], numpy.array(d[y]), lsd[idat], color=col_y[iy], label=y+'_'+l )
#        else :
#            plt.plot( d[x], d[y], lsd[idat], color=col_y[iy], label=y+'_'+l )
#        idat+=1

#plt.xlabel('Time [sec]')
#plt.ylabel('E-lyte Potential [V]')
#plt.legend(loc='best',ncol=3)
