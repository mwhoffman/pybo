"""
Two-dimensional
===============

Demo of Bayesian optimization on two-dimensional test functions with contour
visualizations.

In addition to the previous functionality, this demo features
  - hyperparameter marginalization via SMC,
  - two-dimensional visualization via contour plots, and
  - visualization of the hyperparameter sampling (in Figure 2) via the pygp
    subpackage pygp.plotting.

Try replacing pybo.functions.Branin() with either
  - pybo.functions.Bohachevsky() or
  - pybo.functions.Goldstein()
to experiment with additional two-dimensional built-in test function.
"""

import numpy as np
import matplotlib.pyplot as pl

import pygp
import pygp.plotting as pp
import pybo


def callback(model, bounds, info, x, index, ftrue):
    """
    Plot the current posterior and the index.
    """

    # define grid
    xx0, xx1 = np.meshgrid(
        np.linspace(bounds[0, 0], bounds[0, 1], 200),
        np.linspace(bounds[1, 0], bounds[1, 1], 200))
    xx = np.c_[xx0.flat, xx1.flat]

    # compute true function
    ff = ftrue(xx).reshape(xx0.shape)

    # compute the posterior
    mu, _ = model.posterior(xx)
    mu = mu.reshape(xx0.shape)

    # compute acquisition function
    acq = index(xx).reshape(xx0.shape)

    X = info['x']               # observations
    xbest = info[-1]['xbest']   # recommendations

    fig = pl.figure(1)
    pl.clf()

    pl.subplot(221)
    pl.contour(xx0, xx1, ff, 20, cmap='coolwarm')
    pl.axis(bounds.flatten())
    pl.title('true function')

    pl.scatter(x[0], x[1], marker='o', s=20, color='k', zorder=2)
    pl.scatter(X[:, 0], X[:, 1], facecolors='none', marker='o', s=20, lw=1,
               color='k', zorder=2)
    pl.scatter(xbest[0], xbest[1], marker='+', s=50, lw=2, color='g', zorder=2)

    pl.subplot(222)
    pl.contour(xx0, xx1, mu, 20, cmap='coolwarm')
    pl.axis(bounds.flatten())
    pl.title('posterior mean')
    pl.scatter(xbest[0], xbest[1], marker='+', s=50, lw=2, color='g', zorder=2)

    pl.subplot(223)
    pl.contour(xx0, xx1, acq, 20, cmap='coolwarm')
    pl.axis(bounds.flatten())
    pl.title('acquisition')

    pl.scatter(x[0], x[1], marker='o', s=20, color='k', zorder=2)

    if not np.all(np.isnan(info['xbest'])):
        pl.subplot(224)
        pl.plot(ftrue(info['xbest']), 'g', lw=2, alpha=0.5)
        pl.axis('tight')
        pl.title('regret of recommendation')

    for ax in fig.axes:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    pl.draw()
    pl.show(block=False)

    pl.figure(2)
    pp.plot_samples(model)

    pl.draw()
    pl.show(block=False)


if __name__ == '__main__':
    rng = 0         # random seed

    # define objective function
    noise = 1e-6
    objective = pybo.functions.Branin(noise)

    bounds = objective.bounds       # suggested bounds
    dim = bounds.shape[0]           # dimension of problem
    niter = 30 * dim                # number of iterations (horizon)

    # define kernel to use for modelling,
    kernel = 'matern5'      # 'se' or 'matern{1,3,5}'

    # prescribe initial hyperparameters
    sn = noise      # signal noise
    sf = 1.0        # signal amplitude
    ell = bounds[:, 1] - bounds[:, 0]   # kernel length scale
    mu = 0.0        # prior mean

    # initialize GP
    gp = pygp.BasicGP(sn, sf, ell, mu, kernel=kernel)

    # define priors over GP hyperparameters (required for MCMC and SMC
    # marginalization)
    prior = {'sn': None,
             'sf': pygp.priors.LogNormal(mu=np.log(sf), sigma=1., min=1e-6),
             'ell': pygp.priors.Uniform(ell / 100, ell * 2),
             'mu': pygp.priors.Gaussian(mu, sf)}

    # SMC hyperparameter marginalization (can be replaced with MCMC)
    model = pygp.meta.SMC(gp, prior, n=10, rng=rng)

    info = pybo.solve_bayesopt(
        objective,
        bounds,
        niter=niter,
        policy='ei',                # 'ei', 'pi', 'ucb', or 'thompson'
        init='sobol',               # 'latin', 'sobol', or 'uniform'
        recommender='latent',       # 'incumbent', 'latent', or 'observed
        model=model,
        rng=rng,
        callback=callback)
