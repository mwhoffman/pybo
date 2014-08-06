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
            X = np.array(X, ndmin=2)    # necessary for kernel computation
            indices = [index(model, grad, X) for model in models]
            if grad:
                return tuple([np.sum(_, axis=0) for _ in zip(*indices)])
            else:
                return np.sum(indices, axis=0)
    else:
        def index2(X, grad=False):
            X = np.array(X, ndmin=2)    # necessary for kernel computation
            return index(models, grad, X)
    return index2


def ei(models, fbest, xi=0.0):
    # FIXME (Matt): Before Bobak had added a target value that was either given
    # by `fbest + xi` or was given by the target. This allows us to enter a
    # 'max' value, but I'm not sure that's right. IE: doesn't that mean we're
    # trying to compute the expected improvement over the max? Which should be
    # zero? I took this out because it deserves more thought.
    target = fbest + xi

    # define the index wrt a single model (that should act like a GP model, ie
    # in that it is marginally Gaussian and defines the posterior method).
    def index(model, grad, X):
        ntest, ndim = X.shape
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
            if ntest == 1:
                # this clause is needed to make sure if a single test point is
                # passed, then a (ndmin,)-array is returned as a gradient.
                # (instead of a (1, ndmin)-array which breaks fmin_l_bfgs_b.)
                dei = np.squeeze(dei, axis=0)
            return ei, dei
        else:
            return ei

    return _integrate(models, index)


def pi(models, fbest, xi=0.05):
    target = fbest + xi

    def index(model, grad, X):
        ntest, ndim = X.shape
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
            dpi = dz * pdfz[:, None]
            if ntest == 1:
                # this clause is needed to make sure if a single test point is
                # passed, then a (ndmin,)-array is returned as a gradient.
                # (instead of a (1, ndmin)-array which breaks fmin_l_bfgs_b.)
                dpi = np.squeeze(dpi, axis=0)

            return cdfz, dpi
        else:
            return cdfz

    return _integrate(models, index)


def ucb(models, fbest, delta=0.1, xi=0.2):
    # FIXME -- Bobak: fbest is ignored in this function but is passed by
    # GPPolicy because at the moment it passes (models, fbest) to all
    # policies.
    d = models._kernel.ndim
    a = xi * 2 * np.log(np.pi**2 / 3 / delta)
    b = xi * (4 + d)

    def index(model, grad, X):
        ntest, ndim = X.shape
        posterior = model.posterior(X, grad=grad)
        mu, s2 = posterior[:2]
        beta = a + b * np.log(model.ndata + 1)
        if grad:
            dmu, ds2 = posterior[2:]
            ducb = dmu + 0.5 * np.sqrt(beta / s2[:, None]) * ds2
            if ntest == 1:
                # this clause is needed to make sure if a single test point is
                # passed, then a (ndmin,)-array is returned as a gradient.
                # (instead of a (1, ndmin)-array which breaks fmin_l_bfgs_b.)
                ducb = np.squeeze(ducb, axis=0)
            return mu + np.sqrt(beta * s2), ducb
        else:
            return mu + np.sqrt(beta * s2)
    return _integrate(models, index)


# def thompson(gp, nfeatures=250):
#     return fourier.FourierSample(gp, nfeatures)
