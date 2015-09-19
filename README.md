# pybo

A Python package for modular Bayesian optimization.

This package provides methods for performing optimization of a possibly
noise-corrupted function *f*. In particular this package allows us to place
a prior on the possible behavior of *f* and select points in order to gather
information about the function and its maximum.

[![Build Status][travis-shield]][travis]
[![Coverage Status][coveralls-shield]][coveralls]

[travis]: https://travis-ci.org/mwhoffman/pybo
[coveralls]: https://coveralls.io/r/mwhoffman/pybo
[travis-shield]: https://img.shields.io/travis/mwhoffman/pybo.svg?style=flat
[coveralls-shield]: https://img.shields.io/coveralls/mwhoffman/pybo.svg?style=flat


## Installation

The easiest way to install this package is by running

    pip install -r https://github.com/mwhoffman/pybo/raw/master/requirements.txt

which will install the package and any of its dependencies. Once the package is
installed the included demos can be run directly via python. For example, by
running

    python -m pybo.demos.animated

A full list of demos can be viewed [here](pybo/demos).
