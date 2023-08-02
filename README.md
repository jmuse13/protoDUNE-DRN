# Dynamic Reduction Network for Electron Energy Regression with ProtoDUNE Data

## Dependencies
* H5py
* Numpy
* Pandas
* tqdm
* CUDA 10.1 or higher
* PyTorch 1.8 or higher
* PyTorch Geometric 2.0 or higher
* Python 3.8 or higher

## What this code is
This is a working framework for performing electron energy regression with ProtoDUNE data using a dynamic reduction network. This code will eventually be pubically released, along with data, for benchmarking purposes.

## Running the code
This package comes with three main scripts:

* `make_input_files.py`, formats ROOT files generated using custom C++ code within the ProtoDUNE ecosystem (propietary) into HDF5 files, which are easily and efficiently read into a PyTorch-based framework
* `train.py`, trains the model (GPU compatible)
* `validate.py`, applys the trained model to regress the full dataset (GPU compatible)
* `DynamicReductionNetworkObject.py`, contains the model (GPU compatible)
* In addition, scripts are available to run the various steps via SLURM (partitions specific to the UMN supercomputing cluster)

## Model modes
There are three main modes that one can study the model. More have been investigated but were found to perform similarly to those modes provided.
* `two_d`, input features are wire number, time tick, collected charge on the collection plane (2D image)
* `three_d`, input features are 3D reconstructed space points, collected charge on the collection plane (3D image)
* `three_d_message`, 3 independent networks are trained where each network takes input features of wire number, time tick, collected charge for an induction or collection plane (3 total planes). Message passing between hits in each plane which are connected by a reco space point is performed. The output is a single regressed energy using a linear layer.

## Training script
On the command line, the following arguments can be passed when calling the `train.py` script:

* `--i`, Input file directory
* `--o`, Output directory
* `--e`, Energy target (log or linear)
* `--valid_batch_size`, Validation batch size
* `--train_batch_size`, Train batch size
* `--ep`, Number of epochs
* `--device`, Which device? (GPU or CPU)
* `--max_lr`, Nominal learning rate
* `--min_lr`, Minimum learning rate
* `--mode"`, Training mode

## Validation script
On the command line, the following arguments can be passed when calling the `validate.py` script:

* `--i`, Input file directory
* `--o`, Model directory
* `--e`, Energy target (log or linear)
* `--batch_size`, Batch size
* `--ep`, Trained epoch iteration to apply
* `--device`, Which device? (GPU or CPU)
* `--mode"`, Training mode

## HDF5 maker script
On the command line, the following arguments can be passed when calling the `make_input_files.py` script:

* `--i`, Input file directory
* `--o`, Model directory
* `--mode"`, Training mode
