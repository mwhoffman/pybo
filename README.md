pybo
----

A Python package for modular Bayesian optimization.

This package provides methods for performing optimization of a possibly
noise-corrupted function *f*. In particular this package allows us to place
a prior on the possible behavior of *f* and select points in order to gather
information about the function and its maximum.

[![Build Status](https://travis-ci.org/mwhoffman/pybo.svg)]
(https://travis-ci.org/mwhoffman/pybo)
[![Coverage Status](https://coveralls.io/repos/mwhoffman/pybo/badge.png)]
(https://coveralls.io/r/mwhoffman/pybo)

Installation
============

The easiest way to install this package is by running

    pip install -r https://github.com/mwhoffman/pybo/raw/master/requirements.txt
    pip install git+https://github.com/mwhoffman/pybo.git

The first line installs any dependencies of the package and the second line
installs the package itself. Alternatively the repository can be cloned directly
in order to make any local modifications to the code. In this case the
dependencies can easily be installed by running

    pip install -r requirements.txt

from the main directory. The package itself can be installed by running `python
setup.py` or by symlinking the directory into somewhere on the `PYTHONPATH`.
Once the package is installed the included demos can be run directly via python.
For example, by running

    python -m pybo.demos.beginner

A full list of demos can be viewed [here](pybo/demos).
