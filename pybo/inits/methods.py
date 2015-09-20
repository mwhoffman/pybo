"""
Implementation of methods for sampling initial points.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np

from ..utils import rstate
from .sobol import i4_sobol_generate

__all__ = ['init_middle', 'init_uniform', 'init_latin', 'init_sobol']


def init_middle(bounds):
    """
    Initialize using a single query in the middle of the space.
    """
    return np.mean(bounds, axis=1)[None, :]


def init_uniform(bounds, n=None, rng=None):
    """
    Initialize using `n` uniformly distributed query points. If `n` is `None`
    then use 3D points where D is the dimensionality of the input space.
    """
    rng = rstate(rng)
    bounds = np.array(bounds, ndmin=2, copy=False)
    d = len(bounds)
    n = 3*d if (n is None) else n

    # generate the random values.
    w = bounds[:, 1] - bounds[:, 0]
    X = bounds[:, 0] + w * rng.rand(n, d)

    return X


def init_latin(bounds, n=None, rng=None):
    """
    Initialize using a Latin hypercube design of size `n`. If `n` is `None`
    then use 3D points where D is the dimensionality of the input space.
    """
    rng = rstate(rng)
    bounds = np.array(bounds, ndmin=2, copy=False)
    d = len(bounds)
    n = 3*d if (n is None) else n

    # generate the random samples.
    w = bounds[:, 1] - bounds[:, 0]
    X = bounds[:, 0] + w * (np.arange(n)[:, None] + rng.rand(n, d)) / n

    # shuffle each dimension.
    for i in xrange(d):
        X[:, i] = rng.permutation(X[:, i])

    return X


def init_sobol(bounds, n=None, rng=None):
    """
    Initialize using a Sobol sequence of length `n`. If `n` is `None` then use
    3D points where D is the dimensionality of the input space.
    """
    rng = rstate(rng)
    bounds = np.array(bounds, ndmin=2, copy=False)
    d = len(bounds)
    n = 3*len(bounds) if (n is None) else n

    # generate the random samples.
    skip = rng.randint(100, 200)
    w = bounds[:, 1] - bounds[:, 0]
    X = bounds[:, 0] + w * i4_sobol_generate(d, n, skip).T

    return X
