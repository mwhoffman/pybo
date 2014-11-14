"""
Acquisition functions based on (GP) UCB.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
import numpy as np

# local imports
from ..utils import params

# exported symbols
__all__ = ['UCB']


@params('delta', 'xi')
def UCB(model, delta=0.1, xi=0.2):
    """
    The (GP)UCB acquisition function where `delta` is the probability that the
    upper bound holds and `xi` is a multiplicative modification of the
    exploration factor.
    """
    # FIXME: getting d in this way won't work unless data has been added.
    d = model.data[0].shape[1]
    a = xi * 2 * np.log(np.pi**2 / 3 / delta)
    b = xi * (4 + d)

    def index(X, grad=False):
        posterior = model.posterior(X, grad=grad)
        mu, s2 = posterior[:2]
        beta = a + b * np.log(model.ndata + 1)
        if grad:
            dmu, ds2 = posterior[2:]
            return (mu + np.sqrt(beta * s2),
                    dmu + 0.5 * np.sqrt(beta / s2[:, None]) * ds2)
        else:
            return mu + np.sqrt(beta * s2)

    return index
