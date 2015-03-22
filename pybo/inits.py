"""
Implementation of methods for sampling initial points.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import mwhutils.grid as grid

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
    n = 3*len(bounds) if (n is None) else n
    X = grid.uniform(bounds, n, rng)
    return X


def init_latin(bounds, n=None, rng=None):
    """
    Initialize using a Latin hypercube design of size `n`. If `n` is `None`
    then use 3D points where D is the dimensionality of the input space.
    """
    n = 3*len(bounds) if (n is None) else n
    X = grid.latin(bounds, n, rng)
    return X


def init_sobol(bounds, n=None, rng=None):
    """
    Initialize using a Sobol sequence of length `n`. If `n` is `None` then use
    3D points where D is the dimensionality of the input space.
    """
    n = 3*len(bounds) if (n is None) else n
    X = grid.sobol(bounds, n, rng)
    return X
