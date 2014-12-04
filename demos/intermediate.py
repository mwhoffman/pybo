"""
Demo performing Bayesian optimization on an objective function sampled from a
Gaussian process. This script also demonstrates user-defined visualization via
a callback function that is imported from the advanced demo.

Note that in this demo we are sampling an objective function from a Gaussian
process, but we are not modifying the default GP used within `pybo.solve_bayesopt`.
The default model used internally by `pybo.solve_bayesopt` uses a GP with zero
mean, Matern kernel, and hyperparameters marginalized using MCMC. To modify this
behavior see the advanced demo.

In this demo we also explore additional keyword arguments for solve_bayesopt()
which gives the user control over
  - the selection policy,
  - the initial search grid, and
  - the recommendation strategy.
"""

import numpy as np
import pygp
import pybo

# import callback from advanced demo
import os
import sys
sys.path.append(os.path.dirname(__file__))
from advanced import callback


if __name__ == '__main__':
    rng = 0                                         # random seed
    noise = 1e-6                                    # observation noise
    mean = 0.0                                      # GP mean
    bounds = np.array([3, 5])                       # bounds of the objective

    # define a GP which we will sample an objective from.
    likelihood = pygp.likelihoods.Gaussian(noise)
    kernel = pygp.kernels.Periodic(1, 1, 0.5) + pygp.kernels.SE(1, 1)
    gp = pygp.inference.ExactGP(likelihood, kernel, mean)
    objective = pybo.functions.GPModel(bounds, gp, rng=rng)

    dim = bounds.shape[0]                           # dimension of problem
    niter = 30 * dim                                # number of iterations

    info = pybo.solve_bayesopt(
        objective,
        bounds,
        niter=niter,
        init='latin',                               # initialization strategy
        policy='thompson',                          # exploration policy
        recommender='observed',                     # recommendation strategy
        noisefree=True,
        rng=rng,
        callback=callback)
