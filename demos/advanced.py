"""
Advanced
========

Demo demonstrating Bayesian optimization with hyperparameter sampling. This
script includes the callback function, used for user-defined visualization,
which can be skipped upon first reading. Jump to '__main__' for the meat of the
script.

By default, pybo.solve_bayesopt() uses an MCMC meta-model to estimate
hyperparameter marginalization. In this tutorial we show how to define this
meta-model manually and prescribe our own hyperparameter priors. (Several priors
are implemented in pygp.priors.)

In this script we can also play around with the
  - kernel (SE or Matern),
  - meta-model (MCMC or SMC);
or even pass the GP model to solve_bayesopt() directly in order to fix the
hyperparameters to their initially prescribed setting.
"""

import numpy as np
import matplotlib.pyplot as pl

import pygp
import pybo


def callback(model, bounds, info, x, index, ftrue):
    """
    Plot the current posterior, the index, and the value of the current
    recommendation.
    """

    # define grid
    xmin, xmax = bounds[0]
    xx = np.linspace(xmin, xmax, 500)[:, None]

    # compute true function
    ff = ftrue(xx)

    # compute posterior and quantiles
    mu, s2 = model.posterior(xx)
    lo = mu - 2 * np.sqrt(s2)
    hi = mu + 2 * np.sqrt(s2)

    # compute acquisition function
    acq = index(xx)

    # get y axis plotting range
    ymin, ymax = ff.min(), ff.max()
    ymin -= 0.2 * (ymax - ymin)
    ymax += 0.2 * (ymax - ymin)

    # common plotting kwargs
    kwplot = {'lw': 2,
              'alpha': 0.5}

    fig = pl.figure(1)
    fig.clf()

    pl.subplot(221)
    # plot the posterior
    pl.fill_between(xx.ravel(), lo, hi, color='b', alpha=0.1)
    pl.plot(xx, mu, 'b', **kwplot)
    # plot data
    pl.scatter(info['x'], info['y'], marker='o', facecolor='none', zorder=3)
    # plot true function
    pl.plot(xx, ff, 'k:', **kwplot)
    # plot latest selection and current recommendation
    pl.axvline(x, color='r', **kwplot)
    pl.axvline(info[-1]['xbest'], color='g', **kwplot)
    # decorating...
    pl.axis((xmin, xmax, ymin, ymax))
    pl.ylabel('posterior')

    pl.subplot(223)
    # plot the acquisition function
    pl.fill_between(xx.ravel(), acq.min(), acq, color='r', alpha=0.1)
    pl.axis('tight')
    # plot latest selection
    pl.axvline(x, color='r', **kwplot)
    # decorating...
    pl.xlabel('input')
    pl.ylabel('acquisition')

    pl.subplot(222)
    # plot the value at the current recommendation
    pl.plot(ftrue(info['xbest']), 'g', **kwplot)
    # decorating...
    pl.axis((0, len(info['xbest']), ymin, ymax))
    pl.xlabel('iterations')
    pl.ylabel('value of recommendation')

    for ax in fig.axes:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    pl.draw()
    pl.show(block=False)


if __name__ == '__main__':
    rng = 0         # random seed

    # define the objective function
    noise = 1e-1    # observation noise
    objective = pybo.functions.Gramacy(noise)

    bounds = objective.bounds       # suggested bounds
    dim = bounds.shape[0]           # dimension of problem
    niter = 30 * dim                # number of iterations (horizon)

    # define kernel
    kernel = 'matern3'      # 'se' or 'matern{1,3,5}'

    # prescribe initial hyperparameters
    sn = noise      # signal noise
    sf = 1.0        # signal amplitude
    ell = bounds[:, 1] - bounds[:, 0]   # kernel length scale
    mu = 0.0        # prior mean

    # initialize GP
    gp = pygp.BasicGP(sn, sf, ell, mu, kernel=kernel)

    # define priors over GP hyperparameters (required for MCMC and SMC
    # marginalization)
    prior = {'sn': pygp.priors.Horseshoe(scale=0.1, min=1e-6),
             'sf': pygp.priors.LogNormal(mu=np.log(sf), sigma=1., min=1e-6),
             'ell': pygp.priors.Uniform(ell / 100, ell * 2),
             'mu': pygp.priors.Gaussian(mu, sf)}

    # MCMC hyperparameter marginalization (can be replaced with SMC)
    model = pygp.meta.MCMC(gp, prior, n=10, rng=rng)

    info = pybo.solve_bayesopt(
        objective,
        bounds,
        T=niter,
        policy='ei',                # 'ei', 'pi', 'ucb', or 'thompson'
        init='sobol',               # 'latin', 'sobol', or 'uniform'
        recommender='incumbent',    # 'incumbent', 'latent', or 'observed
        model=model,
        rng=rng,
        callback=callback)
