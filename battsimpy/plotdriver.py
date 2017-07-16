# -*- coding: utf-8 -*-
"""Main driver for plotting simulation results.

Example:
    $ python plotdriver.py /path/to/output/data/
      model_nmc_fvmP2D.conf sim_DCR.conf

"""
import argparse

# battsimpy specific modules
import model


def main(bsp_path, mod_file, sim_file):
    """
    Import the arguments and run the simulation case that has been setup.
    """
    print 'battsimpy path setting  :', bsp_path
    print 'Model file setting      :', mod_file
    print 'Simulation file setting :', sim_file

    # Build the model and perform simulation
    full_1d = model.Model(mod_file, sim_file, bsp_path)
    full_1d.buildmodel()
    print 'Model build complete.\n'
    full_1d.loadresults()
    print 'Simulation data loaded.\n Plotting...\n'
    full_1d.plotresults()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("bsp_path",
                        help="Path to battsimpy installation.")
    parser.add_argument("mod_file",
                        help="Model config file.")
    parser.add_argument("sim_file",
                        help="Simulation config file.")

    args = parser.parse_args()

    bsp_path = args.bsp_path
    mod_file = bsp_path + 'config_files/' + args.mod_file
    sim_file = bsp_path + 'config_files/' + args.sim_file

    main(bsp_path, mod_file, sim_file)
