# -*- coding: utf-8 -*-
"""General model class.

The model class provides a general framework for the supplied sub-models.

The general flow here is as follows:
1) config files are parsed -> __init__()
2) the specified model is initialized -> __init__()
3) the test cases are simulated -> simulate()
4) the results are saved -> saveresults()

Additionally, previous results can be re-loaded and plotted.

Methods:
    buildmodel()
        This initiates the sub-model to be used for the battery simulation.
            e.g., ecm, spm, spm_dist, full_1d, etc.
    buildcase()
        Reads simulation configuration file and sets up the case.
            e.g., Setup the average temperature, delta T, input currents, and
            SOCs to be simulated.
    simulate()
        Simlate all of the cases.
        Calls the sim_single_case() method.
        TODO:
            Provide the ability to simulate as many cases in parallel as
            possible.
    sim_single_case()
        Perform the simulation for the provided case.
    saveresults()
        Save a pickle file of the results_holder variable.
    loadresults()
        Load the results from a previous simulation.
    plot...()
        Plotting methods tailored to the results for certain simlations.
"""

import sys
import numpy
from copy import deepcopy
import itertools
from matplotlib import pyplot as plt
import matplotlib as mpl
import pickle

# battsimpy specific modules
from helper_modules import confreader
from helper_modules import schedreader

# Plot setup
plt.style.use('classic')
FS = 12.
FW = 'bold'
mpl.rc('lines', linewidth=2., color='k')
mpl.rc('font', size=FS, weight=FW, family='Arial')


class Model():
    """
    This is main class for pyBatt models
    """
    def __init__(self, mod_conf_path, sim_conf_path, bsp_path):
        """
        Initialize the model class
        """
        self.mod_conf_path = mod_conf_path
        self.sim_conf_path = sim_conf_path
        self.bsp_path = bsp_path

        self.buildmodel()

    def buildmodel(self):
        """
        Extract all model parameters from associated data files defined in the
        model config file and simulation config file.
        Initialize the specified model.
        """
        # Parse the model and simulation config files.
        mcd = confreader.Reader(self.mod_conf_path)
        scd = confreader.Reader(self.sim_conf_path)

        # Merge the model and simulation config inputs.
        conf_data = mcd.conf_data.copy()
        self.conf_data = conf_data

        for k1 in scd.conf_data.keys():
            if k1 not in conf_data.keys():
                conf_data[k1] = {}
            for k2 in scd.conf_data[k1].keys():
                conf_data[k1][k2] = scd.conf_data[k1][k2]

        # Get the model type that was specified in the conf file.
        model_type = conf_data['MODEL']['MODEL_TYPE']
        self.model_type = model_type

        # Initialize the specified model.
        if 'ecm' in model_type:
            print "Not provided in this version yet."
            sys.exit()
#            from battery_models import ecm
#            self.model = ecm.Ecm(conf_data)
#            self.model.buildpars()  # This creates self.model.pars

        elif 'spm' == model_type:
            print "Not provided in this version yet."
            sys.exit()
#            from battery_models import spm
#            self.model = spm.Spm(conf_data)
#            self.model.buildpars()

        elif 'spm_dist' == model_type:
            print "Not provided in this version yet."
            sys.exit()
#            from battery_models import spm_dist
#            self.model = spm_distributed.Spm_dist(conf_data)
#            self.model.buildpars()

        elif 'pade_1d' == model_type:
            print "Not provided in this version yet."
            sys.exit()
#            import pade_1d
#            self.model = pade_1d.Pade_1D(conf_data)
#            self.model.buildpars()

        elif 'full_1d_fvm_ida' == model_type:
            from battery_models import full_1d_fvm_ida
            self.model = full_1d_fvm_ida.Simulator(conf_data, self.bsp_path)

        elif 'full_1d_dist' == model_type:
            print "Not provided in this version yet."
            sys.exit()
#            from battery_models import full_1d_fvm_ida_dist
#            self.model = full_1d_fvm_ida_dist.Simulator(conf_data,
#                                                        self.bsp_path)

        else:
            sys.exit("MODEL_TYPE not recognized!")

        # Read in test schedule data
        self.sched_path = conf_data['FILEPATHS']['INPUT_DATA_ROOT'] \
            + conf_data['FILEPATHS']['SCHED_PATH']
        self.sched_dat = schedreader.readsched(self.sched_path)

        # Build the simulation case set object
        self.buildcase()

    def buildcase(self,):
        """
        Build an object that contains the case study based on the provided
        simulation config file.

        Two main cases are assumed:
            1) DCR
                This is a study where the DC-Resistance is evaluated using
                short constant current pulses. This is a common case to be
                tested on a real battery, and therefore handy to have built-in
                here for quickly evaluating a model to compare to experimental
                results. A set of initial Depths of Discharge, Temperatures,
                and C-rates are evaluated.
                    e.g.,
                    DOD  = [0.8 , 0.5] # 80% and 50% initial DODs used.
                    T    = [298 , 308] # 25 and 35oC initial temperatures used.
                    Rate = [-1.0,-2.0] # 1C and 2C discharges
                            (neg. here is discharge)

                    This would results in running 8 cases:
                    cases = [[0.8, 298, -1.0], [0.5, 298, -1.0], ...
                             [0.5, 308, -2.0]]
            2) Rate
                Perhaps more common than a DCR test would be to run a set of
                constant current C-rates at several temperatures. This is also
                provided for quick setup.
                A set of initial Temperatures and C-rates are used here. All
                cases for the Rate testing are performed at the same initial
                DOD, and this is the main differentiating factor between a
                DCR and Rate simulation.

            TODO:
                Constant power, dynamic power profiles, constant voltage
        """
        p = self.model.pars

        if 'DCR' in self.model.confdat['SIMULATION']['TEST_TYPE']:
            temps = self.model.confdat['SIMULATION']['TEMP_ARRAY']
            delts = self.model.confdat['SIMULATION']['DELTA_TEMP_ARRAY']
            rates = self.model.confdat['SIMULATION']['RATE_ARRAY']
            dods = self.model.confdat['SIMULATION']['DOD_ARRAY']

            socs = 1.-numpy.array(dods)
            if p.xs[1] < p.xs[0]:
                V_inits = numpy.interp(socs, numpy.flipud(p.xs),
                                       numpy.flipud(p.OCP))
            else:
                V_inits = numpy.interp(socs, p.xs, p.OCP)

            self.cases = list(itertools.product(temps, rates, V_inits, delts))

        elif 'Rate' in self.model.confdat['SIMULATION']['TEST_TYPE']:

            temps = self.model.confdat['SIMULATION']['TEMP_ARRAY']
            delts = self.model.confdat['SIMULATION']['DELTA_TEMP_ARRAY']
            currs = self.model.confdat['SIMULATION']['CURR_ARRAY']
            Vinit = self.model.confdat['SIMULATION']['V_INIT']

            if (len(Vinit) == len(currs)):
                self.cases = list(itertools.product(temps, currs,
                                                    [Vinit[0], ], delts))

                # make each case a list, instead of tuple, to allow case
                # value reassignment
                self.cases = [list(case) for case in self.cases]

                # Assign the correct initial voltage intended for each input
                # current
                for iTemp, I in enumerate(temps):
                    for iCur, T in enumerate(currs):
                        ic = (iTemp-1)*len(currs) + iCur
                        self.cases[ic][2] = Vinit[iCur]

                # convert cases back to tuples
                self.cases = [tuple(case) for case in self.cases]

            elif (len(Vinit) == 1):
                self.cases = list(itertools.product(temps, currs,
                                                    [Vinit[0], ], delts))

            else:
                print "Error with V_INIT setup for Rate study. Should have \
                    only one value, or have a value for each rate in the \
                    study."

    def simulate(self):
        """
        Simulate the model for the given schedule file
        """
        self.results_holder = [object for case in self.cases]

        # Run all cases sequentially.
        for iCase, case in enumerate(self.cases):
            # Run a single case.
            self.results_holder[iCase] = self.sim_single_case(iCase)

            # Iteratively save the results, in case the simualtion crashes
            # along the way.
            self.saveresults()

        # TODO:
            # Provide the ability to run cases in parallel.

    def sim_single_case(self, case_ind):
        """
        Run through each step of the test schedule provided in the simulation
        config file.
        """
        # Setup the initial conditions for this case.
        num_cycs = 1
        case = self.cases[case_ind]

        print '###########'
        print 'case:', case
        print '###########', '\n'

        T = case[0] + 273.15  # [K]
        dT = case[3]

        self.model.V_init = case[2]
        self.model.buildpars()

        print 'T, dT:', T, dT
        self.model.pars.T_amb = T
        self.model.pars.T_avg = T
        self.model.pars.T_dT = dT
        self.model.get_Tvec()

        self.model.buildmodel()
        self.model.buildsim()

        print 'Tvec:', self.model.Tvec

        schd, p = self.sched_dat, self.model.pars

        self.model.schd = schd

        print 'V_init: '+str(self.model.V_init)+'V'

        if schd['StepName'][-1] == 'Repeat':
            num_cycs = schd['StepDuration_sec'][-1]
            num_steps = len(schd['StepNumber']) - 1
        else:
            num_cycs = 1
            num_steps = len(schd['StepNumber'])

        # Build the model results dictionary object
        self.model.build_results_dict(num_steps, num_cycs)
        self.model.t_end_now = 0.0

        # Loop through the test schedule
        pres_step, pres_cyc = 0, 0
        while pres_cyc < num_cycs:
            run_name = 'step'+str(pres_step) + '_repeat'+str(pres_cyc)
            # e.g., step0_repeat0 -> first step and first cycle through the
            # schedule

            p.volt_max = schd['VoltMax'][pres_step]
            p.volt_min = schd['VoltMin'][pres_step]
            p.an_volt_max = schd['AnodeMax'][pres_step]
            p.an_volt_min = schd['AnodeMin'][pres_step]
            p.cat_volt_max = schd['CathodeMax'][pres_step]
            p.cat_volt_min = schd['CathodeMin'][pres_step]
            p.delta_t_max = schd['dt'][pres_step]
            p.volt_lim_on = 0

            print '$$$$$$$$$$$$$$$$$$$'
            print 'Vcutoff:', p.volt_min
            print '$$$$$$$$$$$$$$$$$$$'

            # Handle the first step in the schedule slightly differently
            if pres_step == 0:
                self.model.get_input(schd['InputType'][pres_step],
                                     schd['InputValue'][pres_step])

                t_end_last = 0.0
                tfinal = schd['StepDuration_sec'][pres_step]

                self.model.simulate(tfinal, run_name)

                tend = deepcopy(self.model.t_end_now + t_end_last)

            # All other steps
            else:
                # Get model input voltage or current from the schedule file
                if ('rest' in schd['StepName'][pres_step]
                        or 'Rest' in schd['StepName'][pres_step]):
                    Ifac = 1.0
                else:
                    Ifac = case[1]/schd['InputValue'][pres_step]

                print 'Step Name:', schd['StepName'][pres_step]
                print 'Ifac:', Ifac

                self.model.get_input(schd['InputType'][pres_step],
                                     schd['InputValue'][pres_step]*Ifac)
                if 'Rate' in self.model.confdat['SIMULATION']['TEST_TYPE']:
                    tfinal = schd['StepDuration_sec'][pres_step]/Ifac \
                        + self.model.t_end_now
                    p.delta_t_max = schd['dt'][pres_step]/Ifac
                else:
                    tfinal = schd['StepDuration_sec'][pres_step] \
                        + self.model.t_end_now

                print 'dt max:', p.delta_t_max

                # Setup the time and input arrays for the model
                # These are dealt with differently for the 'DCR' or 'Rate'
                # style schedules.

                # A DCR test may change the rate for several cases, but
                # want to keep the same pulse time for each of those
                # rates, hence Ifac is not necessarily 1, but tfac always
                # will be. But, for a Rate style study, we want full
                # discharge sweeps at each rate, therefore, the time
                # step must increase, if the rate decreases
                # (e.g., 1C discharge for ~3600sec, or 0.1C discharge for
                # 36000sec).
                t_end_last = tend

                # no longer in the initial rest mode (only for pres_step = 0)
#                init_rest_on = 0

                # Execute this next simulation step
                self.model.simulate(tfinal, run_name)

                # Check if we are charging or discharging
                # Negative input current -> charging, (+ -> dchg)
#                charging_on = step_inp[-1]<0

#                inp_last = deepcopy(self.model.inp)
                tend = deepcopy(self.model.t_end_now + t_end_last)
#                step_old = deepcopy(pres_step)
#                cyc_old = deepcopy(pres_cyc)
#                last_run_name = deepcopy(run_name)

            pres_step += 1

            if pres_step == num_steps:
                pres_cyc += 1

        return self.model.results_out

    def saveresults(self):
        """
        Save the resuls using Pickle.
        """
        # Mean temperatures simulated
        nMT = len(self.model.confdat['SIMULATION']['TEMP_ARRAY'])
        # Delta temperatures simulated (only relavant for "_dist" model type)
        nDT = len(self.model.confdat['SIMULATION']['DELTA_TEMP_ARRAY'])

        # Pickle file name for the saved simulation data
        filename = self.model.confdat['SIMULATION']['TEST_TYPE'] + '__' \
            + self.model.confdat['SIMULATION']['SAVE_NAME'] + '__' \
            + self.model.confdat['FILEPATHS']['MODEL_NAME'] + '__' \
            + self.model.confdat['MODEL']['MODEL_TYPE'] + '__' \
            + str(nMT) + 'mTby' + str(nDT) + 'dT.p'

        # Full file path
        filepath = self.model.confdat['FILEPATHS']['OUTPUT_ROOT'] \
            + self.model.confdat['FILEPATHS']['DATE'] + '/' + filename

        pickle.dump(self.results_holder, open(filepath, "wb"), protocol=2)

    def loadresults(self):
        """
        Load the saved resuls using Pickle.
        """
        # Mean temperatures simulated
        nMT = len(self.model.confdat['SIMULATION']['TEMP_ARRAY'])
        # Delta temperatures simulated (only relavant for "_dist" model type)
        nDT = len(self.model.confdat['SIMULATION']['DELTA_TEMP_ARRAY'])

        # Pickle file name for the saved simulation data
        filename = self.model.confdat['SIMULATION']['TEST_TYPE'] + '__' \
            + self.model.confdat['SIMULATION']['SAVE_NAME'] + '__' \
            + self.model.confdat['FILEPATHS']['MODEL_NAME'] + '__' \
            + self.model.confdat['MODEL']['MODEL_TYPE'] + '__' \
            + str(nMT) + 'mTby' + str(nDT) + 'dT.p'

        # Full file path
        filepath = self.model.confdat['FILEPATHS']['OUTPUT_ROOT'] \
            + self.model.confdat['FILEPATHS']['DATE'] + '/' + filename

        self.results_holder = pickle.load(open(filepath, "rb"))

    def plotresults(self):
        """
        Example plots.
        """
        print self.results_holder
        data = self.results_holder[0]
        
        print data['step0_repeat0']

        fig, ax = plt.subplots(1, 2)
        ax_an = ax[0].twinx()
        for stp in range(len(self.sched_dat['StepNumber'])):
            stp_rep = 'step' + str(stp) + '_repeat0'
            if stp == 0:
                # Plot the full cell potential, and the cathode and anode voltages.
                ax[0].plot(data[stp_rep].test_time, data[stp_rep].Volt,
                    '-sb', label='Cell Voltage')
                ax[0].plot(data[stp_rep].test_time, data[stp_rep].Vc,
                    '-sk', label='Cathode Voltage')
                ax_an.plot(data[stp_rep].test_time, data[stp_rep].Va,
                    '-sr', label='Anode Voltage')

                # Plot the input current profile.
                ax[1].plot(data[stp_rep].test_time, data[stp_rep].Cur,
                    '-sb', label='Input current')

                for axi in ax:
                    axi.legend(loc=2)
                    axi.set_xlabel('Test Time [s]')
                ax_an.legend(loc=3)
                ax[0].set_ylabel('Voltage [V]')
                ax[1].set_ylabel('Current [A]')
            else:
                # Plot the full cell potential, and the cathode and anode voltages.
                ax[0].plot(data[stp_rep].test_time, data[stp_rep].Volt,
                    '-sb')
                ax[0].plot(data[stp_rep].test_time, data[stp_rep].Vc,
                    '-sk')
                ax_an.plot(data[stp_rep].test_time, data[stp_rep].Va,
                    '-sr')

                # Plot the input current profile.
                ax[1].plot(data[stp_rep].test_time, data[stp_rep].Cur,
                    '-sb')

        ax[0].set_ylim([3.5, 4.2])
        ax_an.set_ylim([0.0, 0.5])
        ax[1].set_ylim([0.0, 6.0])
        plt.tight_layout()

        plt.show()
