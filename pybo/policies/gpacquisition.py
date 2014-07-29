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
__all__ = ['ei']


def _integrate(models, index):
    if hasattr(models, '__iter__'):
        def index2(X, grad=False):
            indices = [index(model, X, grad) for model in models]
            if grad:
                return tuple([np.sum(_, axis=0) for _ in zip(*indices)])
            else:
                return np.sum(indices, axis=0)
    else:
        def index2(X, grad=False):
            return index(models, X, grad)
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
    def index(model, X, grad=False):
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


# FIXME: just comment out the other acquisition functions for now. NOTE that
# they've also been removed from __all__.


# def pi(gp, xi=0.05):
#     fmax = gp.get_max()[1] if (gp.ndata > 0) else 0
#     def index(X):
#         mu, s2 = gp.posterior(X)
#         mu -= fmax + xi
#         mu /= np.sqrt(s2, out=s2)
#         return mu
#     return index


# def ucb(gp, delta=0.1, xi=0.2):
#     d = gp._kernel.ndim
#     a = xi*2*np.log(np.pi**2 / 3 / delta)
#     b = xi*(4+d)
#     def index(X):
#         mu, s2 = gp.posterior(X)
#         beta = a + b * np.log(gp.ndata+1)
#         return mu + np.sqrt(beta*s2)
#     return index


# def thompson(gp, nfeatures=250):
#     return fourier.FourierSample(gp, nfeatures)
