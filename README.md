# Bayesian Analysis with pyhf using PyMC
The aim of this project is to provide tools for Bayesian analysis of ``pyhf`` models using the Python library ``PyMC``.

## Setup
To set up a Python virtual environment, download the following dependency file [environment.yml](https://github.com/malin-horstmann/bayesian_pyhf/blob/main/environment.yml) and run:

```
conda env create --file environment.yml
```

You can then run
```
pip install --editable .
```
for an editable install of `bayesian_pyhf`.

For a quick example, see [example.ipynb](https://github.com/malin-horstmann/bayesian_pyhf/blob/main/examples/SimpleEx.ipynb).

## Development

### Lock files

To create a fully reproducible environment lock file from the high level `environment.yml`, install [`conda-lock`](https://github.com/conda/conda-lock) and then create a hash-level `conda-lock.yml` lock file with

```
conda-lock lock --file environment.yml --kind lock
```

or more simply, with [`nox`](https://github.com/wntrblm/nox) installed, by running the `nox` default session (which requires Docker)

```
nox
```

An environment can then be created from the lock file either with

```
conda-lock install --name bayesian_pyhf conda-lock.yml
```

or

```
conda env create --file conda-lock.yml
```

See `conda-lock install --help` for additional options.

### Updating the environment

To add new dependencies to the environment definition files simply add the dependencies to the `environment.yml` and then rebuild the lock file.
To update your existing environment from an updated `environment.yml` file use

```
conda env update --name bayesian_pyhf --file environment.yml
```