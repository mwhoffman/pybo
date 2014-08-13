"""
Models which approximate a sample from a given GP.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
import numpy as np

# local imports
from ..utils.random import rstate

# exported symbols
__all__ = ['GPModel']


# FIXME: GPModel is broken now. needs to be fixed.


class GPModel(object):
    """
    Model representing draws from a zero-mean GP prior with the given kernel.
    This model is represented using N features sample from the spectral density
    and in particular by using the same seed (rng) we can use the same sample
    across different runs or different optimizers.

    NOTE: fixing the rng input will fix the function sampled, but for sigma>0
    any noisy data will use numpy's global random state.
    """
    def __init__(self, bounds, gp, N=500, rng=None):
        self.bounds = np.array(bounds, dtype=float, ndmin=2)
        self._rng = rstate(rng)
        self._f = gp.sample_fourier(N, self._rng)
        self._likelihood = gp._likelihood.copy()

    def __call__(self, x):
        return self.get(x)[0]

    def get(self, X):
        return self._likelihood.sample(self.get_f(X), self._rng)

    def get_f(self, X):
        return self._f.get(np.array(X, ndmin=2, copy=False))
