"""
Simple utilities for random number generation.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
import numpy as np

# exported symbols
__all__ = ['rstate', 'random', 'lhsample']


def rstate(rng=None):
    """
    Return a numpy RandomState object. If an integer value is given then a new
    RandomState will be returned with this seed. If None is given then the
    global numpy state will be returned. If an already instantiated state is
    given this will be passed back.
    """
    if rng is None:
        rng = np.random.mtrand._rand
    elif isinstance(rng, int):
        rng = np.random.RandomState(rng)
    elif not isinstance(rng, np.random.RandomState):
        raise ValueError('unknown seed given to rstate')
    return rng


def random(bounds, n, rng=None):
    """
    Sample n points uniformly at random from the specified region, given by
    a list of [(lo,hi), ..] bounds in each dimension.
    """
    # if given a seed or an instantiated RandomState make sure that we use
    # it here, but also within the sample_spectrum code.
    rng = rstate(rng)
    bounds = np.array(bounds, ndmin=2, copy=False)

    # generate the random values.
    d = len(bounds)
    X = bounds[:,0] + rng.rand(n,d) * (bounds[:,1] - bounds[:,0])

    return X


def lhsample(bounds, n, rng=None):
    """
    Sample n points from a latin hypercube within the specified region, given by
    a list of [(lo,hi), ..] bounds in each dimension.
    """
    rng = rstate(rng)
    bounds = np.array(bounds, ndmin=2, copy=False)

    # generate the random samples.
    d = len(bounds)
    X = bounds[:,0] + (bounds[:,1] - bounds[:,0]) * (np.arange(n)[:,None] + rng.rand(n,d)) / n

    # shuffle each dimension.
    for i in xrange(d):
        X[:,i] = rng.permutation(X[:,i])

    return X

