"""
Local gradient-based solver using multiple restarts.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
import numpy as np
import scipy.optimize

# exported symbols
__all__ = ['solve_lbfgsb']


def solve_lbfgsb(func_grad, bounds, ngrid=10000, nbest=10, max=False):
    """
    Compute func on a grid, pick nbest points, and LBFGS from there.

    Args:
        func_grad: function handle that takes an optional `grad` boolean kwarg
            and if `grad=True` returns a tuple of `(function, gradient)`.
            NOTE: this functions is assumed to allow for multiple inputs in
            vectorized form.
        bounds: bounds of the search space.
        ngrid: number of (random) grid points to test initially.
        nbest: number of best points from the initial test points to refine.

    Returns:
        xmin, fmin: location and value or minimizer.
    """

    dim = len(bounds)
    widths = bounds[:, 1] - bounds[:, 0]
    # TODO: The following line could be replaced with a regular grid or a
    # Sobol grid.
    xx = bounds[:, 0] + widths * np.random.rand(ngrid, dim)

    # compute func_grad on points xx
    ff = func_grad(xx, grad=False)
    idx_sorted = np.argsort(ff) if (not max) else np.argsort(ff)[::-1]

    # lbfgsb needs the gradient to be "contiguous", squeezing the gradient
    # protects against func_grads that return ndmin=2 arrays
    def func_grad_(x):
        f, g = func_grad(x[None, :], grad=True)
        if max:
            return -f, -np.squeeze(g)
        return f, np.squeeze(g)

    # TODO: the following can easily be multiprocessed
    result = [scipy.optimize.fmin_l_bfgs_b(func_grad_, x0, bounds=bounds)
              for x0 in xx[idx_sorted[:nbest]]]

    xmin = None
    fmin = np.inf
    for res in result:
        if res[1] < fmin:
            xmin = res[0]
            fmin = res[1]

    if max:
        return xmin, -fmin
    return xmin, fmin
