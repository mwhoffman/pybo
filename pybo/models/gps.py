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
    Model
    """
    def __init__(self, gp, bounds, sigma=0.0, N=200, rng=None):
        self.bounds = np.array(bounds, dtype=float, ndmin=2)
        self.f = FourierSample(gp, N, rng)
        self.sigma = sigma

    def get_data(self, x):
        x = np.array(x, ndmin=2, copy=False)
        y = self.f(x)[0]
        if self.sigma > 0:
            y += np.random.normal(scale=self.sigma)
        return y

    def get_all(self, X):
        X = np.array(X, ndmin=2, copy=False)
        y = self.f(X)
        if self.sigma > 0:
            y += np.random.normal(scale=self.sigma, size=len(y))
        return y
