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
from pygp.extra.fourier import FourierSample

# exported symbols
__all__ = ['GPModel']


class GPModel(object):
    """
    Model representing draws from a zero-mean GP prior with the given kernel.
    This model is represented using N features sample from the spectral density
    and in particular by using the same seed (rng) we can use the same sample
    across different runs or different optimizers.

    NOTE: fixing the rng input will fix the function sampled, but for sigma>0
    any noisy data will use numpy's global random state.
    """
    def __init__(self, bounds, kernel, sigma=0, N=500, rng=None):
        self.bounds = np.array(bounds, dtype=float, ndmin=2)
        self._f = FourierSample(kernel, N, rng)
        self._sigma = sigma

    def __call__(self, x):
        return self.get(x)[0]

    def get(self, X):
        y = self.get_f(X)
        y += np.random.normal(scale=self._sigma, size=len(y)) if (self._sigma > 0) else 0.0
        return y

    def get_f(self, X):
        return self._f(np.array(X, ndmin=2, copy=False))
