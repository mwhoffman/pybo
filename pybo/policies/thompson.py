"""
Implementation of Thompson sampling for continuous spaces.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import collections

import mwhutils.random as random
import mwhutils.linalg as linalg

__all__ = ['Thompson']


class FourierSample(object):
    def __init__(self, model, n, rng=None):
        rng = random.rstate(rng)

        # randomize the feature
        W, a = model._kernel.sample_spectrum(n, rng)

        self._W = W
        self._b = rng.rand(n) * 2 * np.pi
        self._a = np.sqrt(2*a/n)
        self._mean = model._mean.copy()
        self._theta = None

        if model.ndata > 0:
            X, Y = model.data
            Z = np.dot(X, self._W.T) + self._b
            Phi = np.cos(Z) * self._a

            # get the components for regression
            A = np.dot(Phi.T, Phi)
            A = linalg.add_diagonal(A, model._sn2)

            L = linalg.cholesky(A)
            r = Y - self._mean.get_function(X)
            p = np.sqrt(model._sn2) * rng.randn(n)

            self._theta = linalg.solve_cholesky(L, np.dot(Phi.T, r))
            self._theta += linalg.solve_triangular(L, p, True)

        else:
            self._theta = rng.randn(n)

    def __call__(self, x, grad=False):
        if grad:
            F, G = self.get(x, True)
            return F[0], G[0]
        else:
            return self.get(x)[0]

    def get(self, X, grad=False):
        X = np.array(X, ndmin=2, copy=False)
        Z = np.dot(X, self._W.T) + self._b

        F = self._mean.get_function(X)
        F += np.dot(self._a * np.cos(Z), self._theta)

        if not grad:
            return F

        d = (-self._a * np.sin(Z))[:, :, None] * self._W[None]
        G = np.einsum('ijk,j', d, self._theta)

        return F, G


def Thompson(model, n=100, rng=None):
    """
    Implementation of Thompson sampling for continuous models using a finite
    approximation to the kernel matrix with `n` Fourier components.
    """
    if hasattr(model, '__iter__'):
        model = collections.deque(model, maxlen=1).pop()
    return FourierSample(model, n, rng).get
