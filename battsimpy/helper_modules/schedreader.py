# -*- coding: utf-8 -*-
"""Test schedule file reader.

The test schedule file is a .csv formatted file containing a header of
important test variables, and rows of values for each header key.
Each row consists of the step of the test to be carried out.

Example test schedule files:
battsimpy_path/data/ScheduleFiles/*.csv
"""
import csv


def readsched(filepath):
    """
    Execute on reading in and parsing the schedule file
    """
    print 'Schedule file path:', filepath
    try:
        with open(filepath, 'rb') as csvfile:
            reader = csv.reader(csvfile)
            # minus 1 to remove header line count
            nSteps = sum(1 for line in open(filepath)) - 1

            for irow, row in enumerate(reader):
                if irow == 0:
                    # Header
                    header = row
                    dat = dict([(row[akey], range(nSteps))
                               for akey in range(len(row))])
                else:
                    for ikey, akey in enumerate(header):
                        if (akey == 'StepNumber') or (akey == 'VoltControlOn'):
                            dat[akey][irow-1] = int(float(row[ikey]))
                        elif (akey == 'StepName') or (akey == 'InputType'):
                            dat[akey][irow-1] = row[ikey]
                        else:
                            dat[akey][irow-1] = float(row[ikey])

        dat['numSteps'] = nSteps
    except:
        print "Error reading schedule file."

    return dat
