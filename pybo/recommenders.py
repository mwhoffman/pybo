"""
Recommendations.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

from . import solvers

__all__ = ['best_latent', 'best_incumbent', 'best_observed']


def best_latent(model, bounds):
    """
    Given a model return the best recommendation, corresponding to the point
    with maximum posterior mean.
    """
    def mu(X, grad=False):
        """Posterior mean objective function."""
        if grad:
            return model.predict(X, True)[::2]
        else:
            return model.predict(X)[0]
    xgrid, _ = model.data
    xbest, _ = solvers.solve_lbfgs(mu, bounds, xgrid=xgrid)
    return xbest


def best_incumbent(model, _):
    """
    Return a recommendation given by the best latent function value evaluated
    at points seen so far.
    """
    X, _ = model.data
    f, _ = model.predict(X)
    return X[f.argmax()]


def best_observed(model, _):
    """
    Return a recommendation given by the best observed value.
    """
    X, y = model.data
    return X[y.argmax()]
