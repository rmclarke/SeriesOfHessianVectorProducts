This repository contains code for the paper _Series of Hessian-Vector Products
for Saddle-Free Newton Optimisation of Neural Networks_, submitted to TMLR.

This code was co-authored by Ross Clarke (@rmclarke) and Elre Oldewage
(@cravingoxygen).

# Installation
Our dependencies are fully specified in `requirements.txt`, which can be supplied to
`pip` to install the environment.

To setup some local environment variables, you'll need to follow the
instructions in `local_config_template`.

To fix a bug in KFAC-JAX, we use the patch file `kfac_jax.patch`, which you will
need to apply by running the following from the root directory after you have
created the virtual environment:
``` shell
$ patch -p0 -i kfac_jax.patch
```

# Datasets
Our dataset objects are capable of automatically downloading and processing the
necessary datasets to run our code. However, they must first be manually
constructed with the `download=True` argument to trigger this process.

# Running
Training code is controlled with a YAML configuration file, as per the examples
in `configs/`. Brief help text is available on the command line, but the
meanings of each option should be reasonably self-explanatory. Configuration
files in the main `configs/` directory are likely to be most relevant.

Individual runs are commenced by executing `train.py` and passing the desired
configuration file. For example, to run our series algorithm on UCI Energy,
use:
```shell
$ python train.py -c ./configs/uci_energy.yaml ./configs/OursAdaptive.yaml
```
Be sure to specify the dataset file before the algorithm, otherwise default
optimiser settings may overwrite the custom ones required by the algorithm.

To perform hyperparameter tuning, refer
to `parallel_exec.py`, which will create a Ray server to allow for parallelisation
of runs. You will also need to specify the algorithm in use with the `-R` flag,
and may wish to specify the logging root with the `-l` flag and the logging
directory (relative to that root) with the `-g` flag.

We include the optimal configurations selected by our hyperparameter optimiser
under the `configs/` directory.

# Analysis
By default, runs are logged in Tensorboard format to the `./runs/` directory,
where Tensorboard may be used to inspect the results. If desired, a descriptive
name can be appended to a particular execution using the `-n` switch on the
command line.
