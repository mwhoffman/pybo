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

# exported symbols
__all__ = ['ei', 'pi', 'ucb', 'thompson']


def ei(gp, xi=0.0, target=None):
    # target is used instead of fmax + xi whenever the highest possible value
    # is known.

    # TODO -- Bobak: The following code snippet will be copied in pi as well.
    #   Is there a way to maybe consolidate code here, or is it not worth it?

    # if gp is a list, call this function recursively on each element
    if hasattr(gp, '__iter__'):
        indices = [ei(gpi, xi=xi, target=target) for gpi in gp]
        def index(x, grad=False, negate=False):
            uix = [ui(x, grad, negate) for ui in indices]
            if grad:
                ei, gradient = zip(*uix)
                return np.mean(ei, 0), np.mean(gradient, 0)
            return np.mean(uix, 0)
        return index

    # to do on each gp that is passed
    fmax = gp.get_max()[1] if (gp.ndata > 0) else 0
    # NOTE -- Bobak: I think the following is a nice way to unify xi and a
    #   potentially known target.
    target = target if target else fmax + xi

    def index(X, grad=False, negate=False):
        posterior = gp.posterior(X, grad=grad)
        mu, s2 = posterior[:2]
        s = np.sqrt(s2)
        d = mu - target
        z = d / s
        pdfz = ss.norm.pdf(z)
        cdfz = ss.norm.cdf(z)
        ei = d * cdfz + s * pdfz

        if not grad:
            return -ei if negate else ei

        dmu, ds2 = posterior[2:]

        gradient = 0.5 * ds2 / s2
        gradient *= ei - s * z * cdfz
        gradient += cdfz * dmu

        # return ei and grad
        return -ei, -np.array(gradient, ndmin=1) if negate else \
               ei, np.array(gradient, ndmin=1)

    return index


def pi(gp, xi=0.05):
    pass
#     fmax = gp.get_max()[1] if (gp.ndata > 0) else 0
#     def index(X):
#         mu, s2 = gp.posterior(X)
#         mu -= fmax + xi
#         mu /= np.sqrt(s2, out=s2)
#         return mu
#     return index
#
#
def ucb(gp, delta=0.1, xi=0.2):
    pass
#     d = gp._kernel.ndim
#     a = xi*2*np.log(np.pi**2 / 3 / delta)
#     b = xi*(4+d)
#     def index(X):
#         mu, s2 = gp.posterior(X)
#         beta = a + b * np.log(gp.ndata+1)
#         return mu + np.sqrt(beta*s2)
#     return index
#
#
def thompson(gp, nfeatures=250):
    pass
#     return fourier.FourierSample(gp, nfeatures)
