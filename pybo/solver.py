"""
Wrapper around policy-based optimization methods that iteratively queries the
problem and returns the maximizer and additional convergence information.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
import numpy as np

# exported symbols
__all__ = ['solve_policy']


def solve_policy(model, policy, T):
    """
    Find the optimum of a model using a given policy.
    """

    # grab initial design.
    X = policy.get_init()
    n, d = X.shape

    # initialize the datastructure we'll be returning.
    data = np.zeros(T, [('x', np.float, (d,)),
                        ('y', np.float),
                        ('xstar', np.float, (d,)),
                        ('fstar', np.float),
                        ('regret', np.float)])

    # evaluate points found in the initial design. also make sure to only look
    # at the first T points (ie if we're given a very restrictive time horizon).
    for t, x in enumerate(X[:T]):
        y = model(x)
        policy.add_data(x, y)
        data[t] = (x, y, policy.get_best(), np.nan, np.nan)

    # create the datastructure.
    for t in xrange(t+1, T):
        x = policy.get_next()
        y = model(x)
        policy.add_data(x, y)
        data[t] = (x, y, policy.get_best(), np.nan, np.nan)

    if hasattr(model, 'get_f'):
        data['fstar'] = model.get_f(data['xstar'])

    if hasattr(model, 'get_regret'):
        data['regret'] = model.get_regret(data['xstar'])

    return data
