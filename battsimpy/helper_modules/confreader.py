# -*- coding: utf-8 -*-
"""Configuration file reader module.

The configuration files are used for defining the model and simulation input
parameters. This module contains the methods to read the file and output a
dictionary containing those parameters.

-------
Useage:
-------
conf_file_object = confreader.Reader(self.sim_conf_path)
conf_dict = conf_file_object.conf_data

-------------------------------
Config file formatting details:
-------------------------------
There are four main types of inputs that may be provided in the config files:
1) integer, key value used in config file = "integers" or "integer"
    e.g.,
    PARAM1=1
2) list of integers, by using a key value in the config file = "integer_list"
    e.g.,
    PARAM2=1,2,4,5
3) float, key value used in config file = "floats" or "float"
    e.g.,
    PARAM3=2.0
    This gets used mostly for defining coefficients to modify other parameters.
4) float_list, by using a key value in the config file = "float_list"
    e.g.,
    PARAM4=1.0,15.0,25.0
    This gets used for defining input temperatures, rates, DODs, etc. for the
    simulation conditions.
5) string, by using a key value in the config file = "strings" or "string"
    e.g.,
    PARAM5=/path/to/data/
    This is primarily used for file paths.

An example config file:
#------------------------------------------------------------------------------
$ FILEPATHS | value_type=strings
INPUT_DATA_ROOT=/path/to/battsimpy/data/
DATE=20160421
TEST_TYPE=DCR_vs_SOC_and_T
MODEL_NUM=1
PARAMS=Model_Pars
SCHED_PATH=ScheduleFiles/Schedule_DCR_test.csv
$ SIMULATION | value_type=strings
TEST_TYPE=DCR
$ SIMULATION | value_type=float_list
TEMP_ARRAY=25.0
DELTA_TEMP_ARRAY=0.0
# CURR_ARRAY=-1.0
DOD_ARRAY=0.1
RATE_ARRAY=-1.0
V_INIT=4.198
$ MODEL | value_type=strings
MODEL_TYPE=full_1d
ELECTRODE=full
$ MODEL | value_type=integer
CATHODE_ON=1
N_SUBMOD=1
$ MODEL | value_type=float
FOIL_RES=0.0
TAB_RES=0.0
RATE_NOM_CAP=2.06
#------------------------------------------------------------------------------

The $ indicators create a key for the higher level section of the dictionary.
e.g, in this case the first level of the dict is as follows:
conf_dict.keys() = ['FILEPATHS', 'SIMULATION', 'MODEL']
So, the lines with the $ indicator first contain the main key to be used for
those parameters, and then the value_type is defined. This guides the reader
to create the parameters in the specified type format.

The parameters under each high level key are then filled in as a second level
dictionary.

e.g.,
    conf_dict['FILEPATHS'].keys() = ['CODE_ROOT','DATA_ROOT','DATE',
                                     'TEST_TYPE', 'MODEL_NUM', 'PARAMS',
                                     'SCHED_PATH']
    and all of the values to those keys are stored as strings, in this case.
    e.g.,
    conf_dict['FILEPATHS']['MODEL_NUM'] = '1'
"""
import numpy
import sys


class Reader():
    """
    This class is used to read in conf file data for my data processing scripts
    """
    def __init__(self, conf_filepath):
        """
        init function sets the filename and then gets the conf data
        """
        self.file_path = conf_filepath
        self.conf_data = self.main()

    def remove_comments(self, strng):
        """
        Check for inline comments and remove them
        """
        if '#' in strng:
            hashsplit = strng.split('#')
            out = hashsplit[0]
        else:
            out = strng

        return out

    def get_integer(self, conf_file_line):
        """
        read the key value into a list([key, value])
        convert value to an integer
        """
        split_line = conf_file_line.split('=')
        key = split_line[0]
        value = int(self.remove_comments(split_line[1]))
        return [key, value]

    def get_intg_list(self, conf_file_line):
        """
        read the key value into a list([key, value])
        convert value list to a list of integers
        """
        split_line = conf_file_line.split('=')
        key = split_line[0]
        values = [int(float(self.remove_comments(val)))
                  for val in split_line[1].split(',')]
        return [key, values]

    def get_float(self, conf_file_line):
        """
        read the key value into a list([key, value])
        convert value to an integer
        """
        split_line = conf_file_line.split('=')
        key = split_line[0]
        value = float(self.remove_comments(split_line[1]))
        return [key, value]

    def get_float_list(self, conf_file_line):
        """
        read the key value into a list([key, value])
        convert value list to a list of integers
        """
        split_line = conf_file_line.split('=')
        key = split_line[0]
        values = [float(self.remove_comments(val))
                  for val in split_line[1].split(',')]
        return [key, values]

    def get_string(self, conf_file_line):
        """
        read the key value into a list([key, value])
        leave value as a string
        remove '\n' characters
        """
        split_line = conf_file_line.split('=')

        if ',' in split_line[1]:
            split_vals = split_line[1].split(',')
            split_vals[-1] = split_vals[-1][:-1]
            out = [split_line[0], split_vals]
        else:
            out = [split_line[0], split_line[1][:-1]]

        for i, strng in enumerate(out):
            out[i] = self.remove_comments(strng)

        return out

    def get_key(self, conf_file_line):
        """
        Return the key to be used in the data dict
        """
        splitKey = conf_file_line.split(' | ')
        rawkey = splitKey[0]

        nowht = rawkey.replace(" ", "")
        key = nowht.strip('$')

        return key

    def get_type(self, conf_file_line):
        """
        Return the data type to be used for deciding how to sift through
        the data.
        """
        typeRaw = conf_file_line.split(' | ')
        readType = typeRaw[-1].split('=')
        typeFinal = readType[-1].strip()

        if (typeFinal == 'integers') or (typeFinal == 'integer'):
            dat_type = self.get_integer
        elif typeFinal == 'integer_list':
            dat_type = self.get_intg_list
        elif (typeFinal == 'strings') or (typeFinal == 'strings'):
            dat_type = self.get_string
        elif typeFinal == 'float_list':
            dat_type = self.get_float_list
        elif (typeFinal == 'float') or (typeFinal == 'floats'):
            dat_type = self.get_float
        else:
            print "Type:", "'"+typeFinal+"'", "not recognized!"
            sys.exit("Ending program execution.")

        return dat_type

    def get_ln_nums(self, iHdr, hdrInds, nLns, nHdrs, cmntInds):
        """
        Get the line numbers for this header block in the conf file
        """
        if iHdr < nHdrs-1:
            nextInd = hdrInds[iHdr+1]
            presInd = hdrInds[iHdr]

            diff = nextInd-1 - presInd+1
            if diff > 0:
                pre_dat = range(hdrInds[iHdr] + 1, hdrInds[iHdr + 1])
                dat_ln_nums = [val for val in pre_dat if val not in cmntInds]
            else:
                pre_dat = hdrInds[iHdr] + 1
                dat_ln_nums = [val for val in pre_dat if val not in cmntInds]

        elif iHdr == nHdrs - 1:

            nextInd = nLns
            presInd = hdrInds[iHdr]

            diff = nextInd - presInd + 1

            if diff > 0:
                pre_dat = range(hdrInds[iHdr] + 1, nLns)
                dat_ln_nums = [val for val in pre_dat if val not in cmntInds]
            else:
                pre_dat = hdrInds[iHdr] + 1
                dat_ln_nums = [val for val in pre_dat if val not in cmntInds]

        return dat_ln_nums

    def main(self):
        """
        Open the file and parses through it. The information is input
        into a 'key:value' dictionary.
        """
        try:
            with open(self.file_path, 'rb') as conffile:
                tot_dat_lns = 0
                for a_line in conffile:
                    tot_dat_lns += 1
        except:
            print "Error reading model configuration file."

        conf_file = open(self.file_path, 'r')

        raw_dat = range(tot_dat_lns)
        header_array = numpy.zeros(tot_dat_lns)
        comment_array = numpy.zeros(tot_dat_lns)

        linNum, hdrNum, cmntNum = 0, 0, 0
        for a_line in conf_file:
            # Check for header markers
            if '$' in a_line:
                header_array[linNum] = 1
                hdrNum += 1

            # Check for commented out lines
            if (a_line[0] == '#') or (a_line[0:2] == ' #'):
                comment_array[linNum] = 1
                cmntNum += 1
                raw_dat[linNum] = a_line  # fill in dat with each line
            elif ('#' in a_line) and (a_line[0] != '#'):
                # fill in dat with each line
                raw_dat[linNum] = self.remove_comments(a_line)
            else:
                raw_dat[linNum] = a_line  # fill in dat with each line

            linNum += 1

        nHdrs = hdrNum
        hdrInds = [i for (i, val) in enumerate(header_array)
                   if header_array[i] == 1]

#        nCmnt = cmntNum
        cmntInds = [i for (i, val) in enumerate(comment_array)
                    if comment_array[i] == 1]

        keys = ['' for i in range(nHdrs)]
#        dat_types = ['' for i in range(nHdrs)]
        dat_ln_nums = [[] for i in range(nHdrs)]
        funcs = [self.get_float_list for i in range(nHdrs)]

        for iHdr in range(nHdrs):
            line = raw_dat[hdrInds[iHdr]]

            keys[iHdr] = self.get_key(line)
            funcs[iHdr] = self.get_type(line)
            dat_ln_nums[iHdr] = self.get_ln_nums(iHdr, hdrInds, tot_dat_lns,
                                                 nHdrs, cmntInds)

        dat = dict([(key, {}) for key in list(set(keys))])
        for ikey, key in enumerate(keys):
            for iln in dat_ln_nums[ikey]:

                [pulled_key, pulled_vals] = funcs[ikey](raw_dat[iln])
                dat[key][pulled_key] = pulled_vals

        conf_file.close()

        return dat
