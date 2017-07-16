# -*- coding: utf-8 -*-
"""Main driver for a simulation.

This script is used to run a case.

Example:
    $ python testDriver.py /home/username/projects/battsimpy/
      model_nmc_fvmP2D.conf sim_CC.conf

"""
import argparse

# battsimpy specific modules
import model


def main():
    """
    Import the arguments and run the simulation case that has been setup.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("bsp_path",
                        help="Path to battsimpy installation.")
    parser.add_argument("mod_file",
                        help="Assign model using provided config file.")
    parser.add_argument("sim_file",
                        help="Run simulation using provided config file.")

    args = parser.parse_args()

    mod_file = args.bsp_path + 'configs/' + args.mod_file
    sim_file = args.bsp_path + 'configs/' + args.sim_file

    print 'battsimpy path setting  :', args.bsp_path
    print 'Model file setting      :', mod_file
    print 'Simulation file setting :', sim_file

    # Build the model and perform simulation
    full_1d = model.Model(mod_file, sim_file, args.bsp_path)

    # Run simultion
    print 'Running simulation...\n'
    full_1d.simulate()
    print 'Finished simulation.\n'
    full_1d.saveresults()
    print 'Saved the simulation results.'


if __name__ == '__main__':
    main()
