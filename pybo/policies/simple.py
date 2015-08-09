"""
Acquisition functions based on the probability or expected value of
improvement.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import mwhutils.random as random

from ..solvers import solve_lbfgs

__all__ = ['EI', 'PI', 'UCB', 'Thompson', 'OPES']


def EI(model, _, xi=0.0):
    """
    Expected improvement policy with an exploration parameter of `xi`.
    """
    target = model.predict(model.data[0])[0].max() + xi

    def index(X, grad=False):
        """EI policy instance."""
        return model.get_improvement(X, target, grad)

    return index


def PI(model, _, xi=0.05):
    """
    Probability of improvement policy with an exploration parameter of `xi`.
    """
    target = model.predict(model.data[0])[0].max() + xi

    def index(X, grad=False):
        """PI policy instance."""
        return model.get_tail(X, target, grad)

    return index


def Thompson(model, _, n=100, rng=None):
    """
    Thompson sampling policy.
    """
    return model.sample_f(n, rng).get


def UCB(model, _, delta=0.1, xi=0.2):
    """
    The (GP)UCB acquisition function where `delta` is the probability that the
    upper bound holds and `xi` is a multiplicative modification of the
    exploration factor.
    """
    d = model.ndata
    a = xi * 2 * np.log(np.pi**2 / 3 / delta)
    b = xi * (4 + d)

    def index(X, grad=False):
        """UCB policy instance."""
        posterior = model.predict(X, grad=grad)
        mu, s2 = posterior[:2]
        beta = a + b * np.log(model.ndata + 1)
        if grad:
            dmu, ds2 = posterior[2:]
            return (mu + np.sqrt(beta * s2),
                    dmu + 0.5 * np.sqrt(beta / s2[:, None]) * ds2)
        else:
            return mu + np.sqrt(beta * s2)

    return index


def OPES(model, bounds, nopt=50, nfeat=200, rng=None):
    rng = random.rstate(rng)

    if hasattr(model, '__iter__'):
        conditionals = [
            m.condition_fstar(solve_lbfgs(m.sample_f(nfeat, rng).get,
                                          bounds)[1])
            for m in model]
    else:
        conditionals = [
            model.condition_fstar(solve_lbfgs(model.sample_f(nfeat, rng).get,
                                              bounds)[1])
            for _ in xrange(nopt)]

    def index(X, grad=False):
        if not grad:
            H = model.get_entropy(X)
            H -= np.mean([_.get_entropy(X) for _ in conditionals], axis=0)
            return H
        else:
            H, dH = model.get_entropy(X, grad=True)
            parts = zip(*[_.get_entropy(X, True) for _ in conditionals])
            H -= np.mean(parts[0], axis=0)
            dH -= np.mean(parts[1], axis=0)
            return H, dH

    return index
