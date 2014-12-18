"""
Simplest demo performing Bayesian optimization on a one-dimensional test
function. This script also demonstrates user-defined visualization via a
callback function that is imported from the advanced demo.

The `pybo.solve_bayesopt()` function returns `(xrec, info, model)` where

    - `xrec` is a the final recommendation;
    - `info` is a numpy structured array which includes the observed input and
      output data, `info['x']` and `info['y']`, respectively and the
      recommendations made along the way in `info['xbest']`;
    - `model` is the final posterior model.
"""

import numpy as np
import matplotlib.pyplot as pl

import mwhutils
import pybo


bounds = [[0., 2 * np.pi]]              # define the bounds of your search space

def objective(x):
    """
    Objective function

    This function should take a `dim`-dimensional array, where `dim` is the
    dimensionality of the search space. The function should return a scalar,
    possibly noise corrupted, objective value associated with the point `x`.
    """
    noise = 1e-6
    y = -np.cos(x) - np.sin(3 * x)
    y = np.squeeze(y)                   # necessary to return a scalar
    e = noise * np.random.randn()

    return y + e


if __name__ == '__main__':
    xrec, info, model = pybo.solve_bayesopt(
        objective,
        bounds,
        niter=20,
        noisefree=True,                             # remove when adding noise
        rng=0)


    # plotting
    xx = mwhutils.random.grid(bounds, 1000)         # generate (1000,dim) grid
    yy = [objective(xi) for xi in xx]               # function value on grid
    mu, s2 = model.posterior(xx)                    # model posterior on grid...
    lo = mu - 2 * np.sqrt(s2)                       # ... with confidence bands
    hi = mu + 2 * np.sqrt(s2)

    pl.plot(xx, yy, 'k--', label='True')
    pl.plot(info['x'], info['y'], 'ro', label='Observed')
    pl.plot(xx, mu, 'b-', label='Model')
    pl.fill_between(xx.ravel(), lo, hi,
                    color='b', alpha=0.1, label='Confidence')
    pl.vlines(xrec, *pl.ylim(), color='g', label='Recommended')

    pl.xlim(*bounds)
    pl.legend(loc=0)
    pl.show()
