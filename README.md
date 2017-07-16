# battsimpy
A python package for simulating Lithium-ion battery performance. Equivalent
 circuit, Single-Particle, and full Pseudo-2D (P2D) porous electrode models
 are available.

Presently, the P2D model is the only one we are including, and this has
 been verified against the [`LIONSIMBA MATLAB`](http://sisdin.unipv.it/labsisdin/lionsimba.php) package. We have used their finite-volume based approach, and
 would like to refer others to their paper titled: [`LIONSIMBA: A Matlab
 Framework Based on a Finite Volume Model Suitable for Li-Ion Battery
 Design, Simulation, and Control`](http://jes.ecsdl.org/content/163/7/A1192.abstract)

## Author

[`Matthew Klein`](https://www.linkedin.com/in/matt-klein-365b518/)
Contact: mpklein@ucdavis.edu

## Installation

The third-party packages required are as follows: `python2.7`, `numpy`,
`scipy`, `matplotlib`, and `assimulo`.

[`assimulo`](http://www.jmodelica.org/assimulo) is used to provide access to
 the IDA solver in LLNL's Sundials package. While providing a simple interface
 to IDA, it does not provide the use of the sparse direct solvers in IDA,
 which highly limits the compuational speed here. This will be an area for
 improvement for us moving forward.

My preferred method for getting the third-party packages to run `battsimpy`
 is to install Anaconda2 and then conda install assimulo.
Then, after installing Anaconda2:
```
conda install -c chria assimulo=2.9
```
This is the quickest way to get going for me.

## Setup
Two configuration files need to be either generated, or modified from the
 provided examples. The model_...conf and sim_...conf files should be setup.

For simply running the provided model and simulation the only key that needs to
 be changed is the INPUT_DATA_ROOT. This should be:
 /your/path/to/battsimpy/model_parameters/

Change the INPUT_DATA_ROOT key in the model configuration file.

Change the OUTPUT_ROOT and the DATE keys in the simulation configuration file.
The OUTPUT_ROOT should be setup as follows:
/path/for/output/data/

The DATE key is for a subdirectory in your OUTPUT_ROOT to dump the output
 simulation data.
e.g.,
/OUTPUT_ROOT/DATE/ is the full path used for the output simulation data.

## Useage
cd to your local directory for `/path/to/battsimpy/battsimpy/`.

To run an example simulation:
```
python testdriver.py /path/to/battsimpy/ model_conffile.conf sim_conffile.conf
```

Some example plots:
```
python plotdriver.py /path/to/battsimpy/ model_conffile.conf sim_conffile.conf
```

## More detailed setup
We are working to provide a more detailed User Manual to explain how to setup
 the full set of model parameters and configuration files from scratch for
 users that need to take full advantage of this.
