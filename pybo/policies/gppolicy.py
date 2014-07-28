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

# not "exactly" local, but...
import pygp

# local imports
from ._base import Policy
from ._direct import solve_direct
from . import gpacquisition

# exported symbols
__all__ = ['GPPolicy']


#===============================================================================
# define dictionaries containing functions that can be used for various parts of
# the meta policy

POLICIES = dict((f, getattr(gpacquisition, f)) for f in gpacquisition.__all__)
SOLVERS = dict(direct=solve_direct)
MARGINALS = dict(fixed=lambda gp: gp)


#===============================================================================
# define the meta policy.

class GPPolicy(Policy):
    def __init__(self,
                 bounds,
                 kernel='Matern',
                 solver='direct',
                 policy='ei',
                 marginal='fixed'):

        # initialize the bounds and grab the function objects that will be used
        # as part of the meta policy.
        self._bounds = np.array(bounds, dtype=float, ndmin=2)
        self._solver = SOLVERS[solver]
        self._policy = POLICIES[policy]
        self._marginal = MARGINALS[marginal]

        # FIXME: do something here for specifying sane initial hyperparameters.
        sn = 0.5
        sf = 1.0
        ell = (self._bounds[:,1] - self._bounds[:,0]) / 10

        # FIXME: sane priors?
        self._prior = dict(
            sn =pygp.priors.Uniform(0.01,  1.0),
            sf =pygp.priors.Uniform(0.01, 10.0),
            ell=pygp.priors.Uniform(0.01,  1.0))

        # initialize the "model"
        self._gp = pygp.BasicGP(sn, sf, ell)

    def add_data(self, x, y):
        self._gp.add_data(x, y)
        self._index = self._policy(self._marginal(self._gp))

    def get_next(self):
        if self._gp.ndata == 0:
            xnext = self._bounds[:,1] - self._bounds[:,0]
            xnext /= 2
            xnext += self._bounds[:,0]
        else:
            xnext, _ = self._solver(lambda x: -self._index(x), self._bounds)
        return xnext

    def get_best(self):
        xmax, _ = self._gp.get_max()
        return xmax
