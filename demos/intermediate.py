"""
Intermediate
============

Demo performing Bayesian optimization on an objective function sampled from a
prescribed GP prior. This script also demonstrates user-defined visualization
via a callback function that is imported from the advanced demo.

This demo shows how to define likelihood and covariance functions from which a
Gaussian process (GP) model is then initialized. Note, however, that in this
demo the GP is only used to sample a test function from. The default model used
internally by pybo.solve_bayesopt() is an MCMC meta-model which implements
hyperparameter marginalization.

In this demo we also explore additional keyword arguments for solve_bayesopt()
which gives the user control over
  - the selection policy,
  - the initial search grid, and
  - the recommendation strategy.
"""

import numpy as np

import pygp
import pybo

import os
import sys
sys.path.append(os.path.dirname(__file__))
from advanced import callback


if __name__ == '__main__':
    rng = 0         # random seed
    noise = 1e-6    # observation noise
    mean = 0.0      # GP mean

    # define likelihood and covariance kernel
    likelihood = pygp.likelihoods.Gaussian(noise)
    kernel = (pygp.kernels.Periodic(1, 1, 0.5) +    # with pygp, one can easily
              pygp.kernels.SE(1, 1))                # compose built-in kernels

    # define GP from which the objective function will be sampled
    # Note: the following is *not* the GP that is used internally by
    # solve_bayesopt(). See demos/advanced.py for a demo on how to prescribe
    # that model.
    gp = pygp.inference.ExactGP(likelihood, kernel, mean)

    # sample objective function given bounds and the GP
    bounds = np.array([3, 5])
    objective = pybo.functions.GPModel(bounds, gp, rng=rng)

    dim = bounds.shape[0]       # dimension of problem
    niter = 30 * dim            # number of iterations (horizon)

    info = pybo.solve_bayesopt(
        objective,
        bounds,
        T=niter,
        policy='thompson',          # 'ei', 'pi', 'ucb', or 'thompson'
        init='latin',               # 'latin', 'sobol', or 'uniform'
        recommender='observed',     # 'incumbent', 'latent', or 'observed'
        noisefree=True,
        rng=rng,
        callback=callback)
