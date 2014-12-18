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

import pybo


bounds = [0., 2 * np.pi]                # define the bounds of your search space

def objective(x):                       # define your objective function
    noise = 1e-6
    y = -np.cos(x) - np.sin(3 * x)
    e = noise * np.random.randn()

    return np.squeeze(y + e)


if __name__ == '__main__':
    xrec, info, model = pybo.solve_bayesopt(
        objective,
        bounds,
        niter=20,
        noisefree=True,                             # remove when adding noise
        rng=0)

    # plotting
    xx = np.linspace(*bounds, num=1000)             # grid for plotting
    yy = objective(xx)                              # function value on grid
    mu, s2 = model.posterior(xx[:, None])           # model posterior on grid...
    lo = mu - 2 * np.sqrt(s2)                       # ... with confidence bands
    hi = mu + 2 * np.sqrt(s2)

    pl.plot(xx, yy, 'k--', label='True')
    pl.plot(info['x'], info['y'], 'ro', label='Observed')
    pl.plot(xx, mu, 'b-', label='Model')
    pl.fill_between(xx, lo, hi, color='b', alpha=0.1, label='Confidence')
    pl.vlines(xrec, *pl.ylim(), color='g', label='Recommended')

    pl.xlim(*bounds)
    pl.legend(loc=0)
    pl.show()
