# battsimpy
A python package for simulating Lithium-ion battery performance. Equivalent
 circuit, Single-Particle, and full Pseudo-2D models are available.

## Author

[`Matthew Klein`](https://www.linkedin.com/in/matt-klein-365b518/)

## Installation

The third-party packages required are as follows: `python2.7`, `numpy`,
`scipy`, `matplotlib`, and `assimulo`.

[`assimulo`](http://www.jmodelica.org/assimulo) is used to provide access to
 the IDA sovler in LLNL's Sundials package. While providing a simple interface
 to IDA, it does not provide the use of the sparse direct solvers in IDA,
 which highly limits the compuational speed here. This will be an area for
 improvement for us moving forward.

My preferred method for setup to use `battsimpy` is to simply install Anaconda2
 and then conda install assimulo. After installing Anaconda2:
```
conda install -c chria assimulo=2.9
```
This is the quickest way to get going for me.

## Setup

Two configuration files need to be either generated, or modified from the
 provided examples. The configuration files are located in the config_files
 directory. The model_...cfg and sim_...cfg files.

## Useage


