"""
Acquisition functions for Bayesian optimization with GPs.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
import numpy as np
import scipy.stats as ss
import functools
import operator

# exported symbols
__all__ = ['ei', 'pi', 'ucb']


def _integrate(models, index):
    if hasattr(models, '__iter__'):
        def index2(X, grad=False):
            indices = [index(model, grad, X) for model in models]
            if grad:
                return tuple([np.sum(_, axis=0) for _ in zip(*indices)])
            else:
                return np.sum(indices, axis=0)
    else:
        def index2(X, grad=False):
            return index(models, grad, X)
    return index2


def _get_best(models):
    if not isinstance(models, list):
        models = [models]
    f = np.mean([gp.posterior(gp._X)[0] for gp in models], axis=0)
    j = f.argmax()
    fbest = f[j]
    xbest = models[0]._X[j]
    return fbest, xbest


def ei(models, xi=0.0):
    fbest = _get_best(models)[0]
    target = fbest + xi

    # define the index wrt a single model (that should act like a GP model, ie
    # in that it is marginally Gaussian and defines the posterior method).
    def index(model, grad, X):
        posterior = model.posterior(X, grad=grad)
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
            dei = 0.5 * ds2 / s2[:,None]
            dei *= (ei - s * z * cdfz)[:,None]
            dei += cdfz[:,None] * dmu
            return ei, dei
        else:
            return ei

    return _integrate(models, index)


def pi(models, xi=0.05):
    fbest = _get_best(models)[0]
    target = fbest + xi

    def index(model, grad, X):
        posterior = model.posterior(X, grad=grad)
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

    return _integrate(models, index)


def ucb(models, delta=0.1, xi=0.2):
    def index(model, grad, X):
        d = model._kernel.ndim
        a = xi * 2 * np.log(np.pi**2 / 3 / delta)
        b = xi * (4 + d)

        posterior = model.posterior(X, grad=grad)
        mu, s2 = posterior[:2]
        beta = a + b * np.log(model.ndata + 1)
        if grad:
            dmu, ds2 = posterior[2:]
            return mu + np.sqrt(beta * s2), dmu + 0.5 * np.sqrt(beta / s2[:, None]) * ds2
        else:
            return mu + np.sqrt(beta * s2)

    # FIXME: while this can be implemented, it is not correct in that it does
    # not form a true UCB when integrating over the hyperparameters.
    return _integrate(models, index)


# def thompson(gp, nfeatures=250):
#     return fourier.FourierSample(gp, nfeatures)
