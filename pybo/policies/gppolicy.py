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
INFERENCE = dict(fixed=lambda gp: gp)


#===============================================================================
# define the meta policy.

class GPPolicy(Policy):
    def __init__(self,
                 bounds,
                 noise=None,
                 kernel='Matern3',
                 solver='direct',
                 policy='ei',
                 inference='fixed'):

        # initialize the bounds and grab the function objects that will be used
        # as part of the meta policy.
        self._bounds = np.array(bounds, dtype=float, ndmin=2)
        self._solver = SOLVERS[solver]
        self._policy = POLICIES[policy]
        self._inference = INFERENCE[inference]

        # initialize the GP.
        if isinstance(kernel, pygp.kernels._base.Kernel):
            # FIXME: this should be generalized at some point to more general
            # likelihood models.
            sn = 0.5 if (noise is None) else noise
            likelihood = pygp.likelihoods.Gaussian(sn)
            self._gp = pygp.inference.ExactGP(likelihood, kernel)

        else:
            # come up with some sane initial hyperparameters.
            sn = 0.5 if (noise is None) else noise
            sf = 1.0
            ell = (self._bounds[:,1] - self._bounds[:,0]) / 10
            self._gp = pygp.BasicGP(sn, sf, ell, kernel=kernel)

    def add_data(self, x, y):
        self._gp.add_data(x, y)
        self._marginal = self._inference(self._gp)
        self._index = self._policy(self._marginal)

    def get_next(self):
        if self._gp.ndata == 0:
            xnext = self._bounds[:,1] - self._bounds[:,0]
            xnext /= 2
            xnext += self._bounds[:,0]
        else:
            xnext, _ = self._solver(lambda x: -self._index(x), self._bounds)
        return xnext

    def get_best(self):
        return gpacquisition._get_best(self._marginal)[1]
