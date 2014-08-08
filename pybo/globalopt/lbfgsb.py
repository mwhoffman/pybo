from __future__ import division
import numpy as np
import scipy.optimize


__all__ = ['solve_lbfgsb']


def solve_lbfgsb(func_grad, bounds, ngrid=10000, nbest=10, args=()):
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
    idx_sorted = np.argsort(ff)

    args = list(args) + ['grad=True']
    # TODO: the following can easily be multiprocessed
    result = [scipy.optimize.fmin_l_bfgs_b(
                  func_grad, x0,
                  bounds=bounds,
                  args=args)
              for x0 in xx[idx_sorted[:nbest]]]

    xmin = None
    fmin = np.inf
    for res in result:
        if res[1] < fmin:
            xmin = res[0]
            fmin = res[1]

    return xmin, fmin