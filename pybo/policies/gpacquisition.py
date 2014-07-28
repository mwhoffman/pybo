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


def ei(gp, xi=0.0):
    fmax = gp.get_max()[1] if (gp.ndata > 0) else 0
    def index(X):
        mu, s2 = gp.posterior(X)
        s = np.sqrt(s2, out=s2)
        d = mu - fmax - xi
        z = d / s
        return d*ss.norm.cdf(z) + s*ss.norm.pdf(z)
    return index


def pi(gp, xi=0.05):
    fmax = gp.get_max()[1] if (gp.ndata > 0) else 0
    def index(X):
        mu, s2 = gp.posterior(X)
        mu -= fmax + xi
        mu /= np.sqrt(s2, out=s2)
        return mu
    return index


def ucb(gp, delta=0.1, xi=0.2):
    d = gp._kernel.ndim
    a = xi*2*np.log(np.pi**2 / 3 / delta)
    b = xi*(4+d)
    def index(X):
        mu, s2 = gp.posterior(X)
        beta = a + b * np.log(gp.ndata+1)
        return mu + np.sqrt(beta*s2)
    return index


def thompson(gp, nfeatures=250):
    return fourier.FourierSample(gp, nfeatures)
