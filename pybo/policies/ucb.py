"""
Acquisition functions based on (GP) UCB.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np

__all__ = ['UCB']


def UCB(model, delta=0.1, xi=0.2):
    """
    The (GP)UCB acquisition function where `delta` is the probability that the
    upper bound holds and `xi` is a multiplicative modification of the
    exploration factor.
    """
    d = model.ndata
    a = xi * 2 * np.log(np.pi**2 / 3 / delta)
    b = xi * (4 + d)

    def index(X, grad=False):
        posterior = model.get_posterior(X, grad=grad)
        mu, s2 = posterior[:2]
        beta = a + b * np.log(model.ndata + 1)
        if grad:
            dmu, ds2 = posterior[2:]
            return (mu + np.sqrt(beta * s2),
                    dmu + 0.5 * np.sqrt(beta / s2[:, None]) * ds2)
        else:
            return mu + np.sqrt(beta * s2)

    return index
