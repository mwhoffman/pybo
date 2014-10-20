"""
Implementation of methods for sampling initial points.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
import numpy as np

# local imports
from ..utils import ldsample

# exported symbols
__all__ = ['init_middle', 'init_uniform', 'init_latin', 'init_sobol']


def init_middle(bounds):
    return np.mean(bounds, axis=1)[None,:]


def init_uniform(bounds):
    n = 3*len(bounds)
    X = ldsample.random(bounds, n)
    return X


def init_latin(bounds):
    n = 3*len(bounds)
    X = ldsample.latin(bounds, n)
    return X


def init_sobol(bounds):
    n = 3*len(bounds)
    X = ldsample.sobol(bounds, n)
    return X
