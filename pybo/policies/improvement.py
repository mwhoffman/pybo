"""
Acquisition functions based on the probability or expected value of
improvement.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import scipy.stats as ss

__all__ = ['EI', 'PI']


def _integrate(index, models):
    """
    Helper method which integrates the given index function over the given
    models. Here `models` can be any iterable object where each element
    returned by the iterator could have been passed to the index object itself.
    """
    def index2(X, grad=False):
        indices = [index(X, grad, model) for model in models]
        if grad:
            return tuple([np.sum(_, axis=0) for _ in zip(*indices)])
        else:
            return np.sum(indices, axis=0)
    return index2


def EI(model, bounds, xi=0.0):
    """
    Expected improvement policy with an exploration parameter of `xi`.
    """
    X, _ = model.data
    f, _ = model.get_posterior(X)
    target = f.max() + xi

    # define the index wrt a single model (that should act like a GP model, ie
    # in that it is marginally Gaussian and defines the posterior method).
    def index(X, grad=False, model=model):
        posterior = model.get_posterior(X, grad=grad)
        mu, s2 = posterior[:2]
        s = np.sqrt(s2)
        d = mu - target
        z = d / s
        pdfz = ss.norm.pdf(z)
        cdfz = ss.norm.cdf(z)
        ei = d * cdfz + s * pdfz

        if grad:
            # get the derivative of ei. The mu/s2/etc. components are vectors
            # collecting n scalar points, whereas dmu and ds2 are (n,d)-arrays.
            # The indexing tricks just interpret the "scalar" quantities as
            # (n,1)-arrays so that we can use numpy's broadcasting rules.
            dmu, ds2 = posterior[2:]
            dei = 0.5 * ds2 / s2[:, None]
            dei *= (ei - s * z * cdfz)[:, None]
            dei += cdfz[:, None] * dmu
            return ei, dei
        else:
            return ei

    if hasattr(model, '__iter__'):
        return _integrate(index, model)
    else:
        return index


def PI(model, bounds, xi=0.05):
    """
    Probability of improvement policy with an exploration parameter of `xi`.
    """
    X, _ = model.data
    f, _ = model.get_posterior(X)
    target = f.max() + xi

    def index(X, grad=False, model=model):
        posterior = model.get_posterior(X, grad=grad)
        mu, s2 = posterior[:2]
        s = np.sqrt(s2)
        d = mu - target
        z = d / s
        cdfz = ss.norm.cdf(z)

        if grad:
            # get the derivative of pi. The mu/s2/etc. components are vectors
            # collecting n scalar points, whereas dmu and ds2 are (n,d)-arrays.
            # The indexing tricks just interpret the "scalar" quantities as
            # (n,1)-arrays so that we can use numpy's broadcasting rules.
            dmu, ds2 = posterior[2:]
            dz = dmu / s[:, None] - 0.5 * ds2 * z[:, None] / s2[:, None]
            pdfz = ss.norm.pdf(z)

            return cdfz, dz * pdfz[:, None]
        else:
            return cdfz

    if hasattr(model, '__iter__'):
        return _integrate(index, model)
    else:
        return index
