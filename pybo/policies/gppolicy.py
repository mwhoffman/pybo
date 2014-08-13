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
import pygp.meta

# local imports
from ._base import Policy
from .. import globalopt
from . import gpacquisition

# exported symbols
__all__ = ['GPPolicy']


#===============================================================================
# define dictionaries containing functions that can be used for various parts of
# the meta policy

def _make_dict(module, lstrip='', rstrip=''):
    def generator():
        for fname in module.__all__:
            f = getattr(module, fname)
            fname = fname[len(lstrip):] if fname.startswith(lstrip) else fname
            fname = fname[::-1][len(rstrip):][::-1] if fname.endswith(rstrip) else fname
            fname = fname.lower()
            yield fname, f
    return dict(generator())

MODELS   = _make_dict(pygp.meta)
SOLVERS  = _make_dict(globalopt, lstrip='solve_')
POLICIES = _make_dict(gpacquisition)


#===============================================================================
# define the meta policy.

class GPPolicy(Policy):
    def __init__(self, bounds, noise,
                 kernel='Matern3',
                 solver='direct',
                 policy='ei',
                 inference='fixed',
                 prior=None):

        # make sure the bounds are a 2d-array.
        bounds = np.array(bounds, dtype=float, ndmin=2)

        if isinstance(kernel, str):
            # FIXME: come up with some sane initial hyperparameters.
            sn = noise
            sf = 1.0
            ell = (bounds[:,1] - bounds[:,0]) / 10
            gp = pygp.BasicGP(sn, sf, ell, kernel=kernel)

            if prior is None:
                # FIXME: this is not necessarily a good default prior, but it's
                # useful for testing purposes for now.
                prior = dict(
                    sn =pygp.priors.Uniform(0.01, 1.0),
                    sf =pygp.priors.Uniform(0.01, 5.0),
                    ell=pygp.priors.Uniform(0.01, 1.0))

        if inference is not 'fixed' and prior is None:
            raise Exception('a prior must be specified for models with'
                            'hyperparameter inference and non-default kernels')

        # save all the bits of our meta-policy.
        self._bounds = bounds
        self._solver = SOLVERS[solver]
        self._policy = POLICIES[policy]
        self._model = gp if (inference is 'fixed') else MODELS[inference](gp, prior)

    def add_data(self, x, y):
        self._model.add_data(x, y)

    def get_next(self, return_index=False):
        index = self._policy(self._model)
        xnext, _ = self._solver(index, self._bounds, max=True)
        return (xnext, index) if return_index else xnext

    def get_best(self):
        def objective(X, grad=False):
            return self._model.posterior(X, True)[::2] if grad else \
                   self._model.posterior(X)[0]
        Xtest, _ = self._model.data
        xbest, _ = globalopt.solve_lbfgs(objective, self._bounds, xx=Xtest, max=True)
        return xbest
