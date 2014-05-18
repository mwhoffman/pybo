"""
Wrapper class for simple GP-based policies whose acquisition functions are
simple functions of the posterior sufficient statistics.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
import numpy as np
import scipy.stats as ss

# local imports
from ._base import Policy
from ._direct import solve_direct

# not "exactly" local, but...
from pygp import BasicGP
from pygp.extra.fourier import FourierSample

# exported symbols
__all__ = ['GPPolicy']


#===============================================================================
# we'll first create a dictionary which will map strings to acquisition
# functions and create the decorator _register to put them into this dict. NOTE:
# this does absolutely nothing to the function decorated.

ACQUISITION_FUNCTIONS = dict()

def _register(f):
    ACQUISITION_FUNCTIONS[f.__name__] = f
    return f


#===============================================================================
# definition of "simple" acquisition functions. all of the following functions
# act sort of like constructors for the acquisition function. they take a GP
# object and some optional set of parameters and return an index function which
# can then be optimized.

@_register
def gpei(gp, xi=0.0):
    fmax = gp.xmax()[1] if (gp.ndata > 0) else 0
    def index(X):
        mu, s2 = gp.posterior(X)
        s = np.sqrt(s2, out=s2)
        d = mu - fmax - xi
        z = d / s
        return d*ss.norm.cdf(z) + s*ss.norm.pdf(z)
    return index


@_register
def gppi(gp, xi=0.05):
    fmax = gp.xmax()[1] if (gp.ndata > 0) else 0
    def index(X):
        mu, s2 = gp.posterior(X)
        mu -= fmax + xi
        mu /= np.sqrt(s2, out=s2)
        return mu
    return index


@_register
def gpucb(gp, delta=0.1, xi=0.2):
    d = gp._kernel.ndim
    a = xi*2*np.log(np.pi**2 / 3 / delta)
    b = xi*(4+d)
    def index(X):
        mu, s2 = gp.posterior(X)
        beta = a + b * np.log(gp.ndata+1)
        return mu + np.sqrt(beta*s2)
    return index


@_register
def thompson(gp, nfeatures=250):
    return FourierSample(gp, nfeatures)


#===============================================================================
# define the meta policy.

class GPPolicy(Policy):
    def __init__(self, bounds, sn, sf, ell, acq='gpucb', **extra):
        if acq not in ACQUISITION_FUNCTIONS:
            raise RuntimeError('unknown acquisition function')

        self._bounds = np.array(bounds, dtype=float, ndmin=2)
        self._gp = BasicGP(sn, sf, np.array(ell, ndmin=1))
        self._extra = extra

        # _acq generates _index every time we add data. technically we don't
        # really need to save the index, but it's useful for
        # debugging/visualization purposes.
        self._acq = ACQUISITION_FUNCTIONS[acq]
        self._index = self._acq(self._gp, **self._extra)

    def add_data(self, x, y):
        self._gp.add_data(x, y)
        self._index = self._acq(self._gp, **self._extra)

    def get_next(self):
        xmax, _ = solve_direct(lambda x: -self._index(x), self._bounds)
        return xmax

    def get_best(self):
        xmax, _ = gp.xmax()
        return xmax
