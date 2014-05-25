# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
import numpy as np

# exported symbols
__all__ = ['optimize']


def optimize(model, policy, T, seed=None):
    """
    Find the optimum of a model using a given policy. Run the given policy for T
    iterations. If seed is given seed numpy's global random state with the given
    seed.
    """
    if seed is not None:
        np.random.seed(seed)

    # grab the first point so we can initialize our datastructure.
    x = policy.get_next()

    # create the datastructure.
    d = len(x)
    data = np.zeros(T, [('x', np.float, (d,)),
                        ('y', np.float),
                        ('xstar', np.float, (d,)),
                        ('fstar', np.float),
                        ('regret', np.float)])

    for t in xrange(T):
        y = model(x)
        policy.add_data(x, y)
        data[t] = (x, y, policy.get_best(), np.nan, np.nan)
        x = policy.get_next() if (t < T-1) else None

    if hasattr(model, 'get_f'):
        data['fstar'] = model.get_f(data['xstar'])

    if hasattr(model, 'get_regret'):
        data['regret'] = model.get_regret(data['xstar'])

    return data


