"""
Demo performing Bayesian optimization on an objective function sampled from a
Gaussian process. This script also demonstrates user-defined visualization via
a callback function that is imported from the advanced demo.

Note that in this demo we are sampling an objective function from a Gaussian
process. We are not, however, modifying the default GP used internally by
`pybo.solve_bayesopt`. The default model used within `pybo.solve_bayesopt` is a
GP with constant mean, Matern 5 kernel, and hyperparameters marginalized using
MCMC. To modify this behavior see the advanced demo.

In this demo we also explore the following additional Bayesian optimization
modules that can be user-defined:
  - the initial search grid,
  - the selection policy,
  - the recommendation strategy, and
  - composite kernels (a `pygp` feature).
"""

import numpy as np
import matplotlib.pyplot as pl

import pygp
import pybo


if __name__ == '__main__':
    rng = 0                                         # random seed
    bounds = np.array([3, 5])                       # bounds of search space
    dim = bounds.shape[0]                           # dimension of space

    # define a GP which we will sample an objective from.
    likelihood = pygp.likelihoods.Gaussian(sigma=1e-6)
    kernel = pygp.kernels.Periodic(1, 1, 0.5) + pygp.kernels.SE(1, 1)
    gp = pygp.inference.ExactGP(likelihood, kernel, mean=0.0)
    objective = pybo.functions.GPModel(bounds, gp, rng=rng)

    xrec, info, model = pybo.solve_bayesopt(
        objective,
        bounds,
        niter=30*dim,
        init='latin',                               # initialization policy
        policy='thompson',                          # exploration policy
        recommender='observed',                     # recommendation policy
        noisefree=True,
        rng=rng)

    # plotting
    xx = np.linspace(*bounds, num=1000)             # grid for plotting
    yy = objective.get(xx[:, None])                 # function value on grid
    mu, s2 = model.posterior(xx[:, None])           # model posterior on grid...
    lo = mu - 2 * np.sqrt(s2)                       # ... with confidence bands
    hi = mu + 2 * np.sqrt(s2)

    pl.plot(xx, yy.ravel(), 'k--', label='True')
    pl.plot(info['x'], info['y'], 'ro', label='Observed')
    pl.vlines(xrec, *pl.ylim(), color='g', label='Recommended')
    pl.plot(xx, mu, 'b-', label='Model')
    pl.fill_between(xx, lo, hi, color='b', alpha=0.1, label='Confidence')

    pl.xlim(*bounds)
    pl.legend(loc=0)
    pl.show()