"""
Demo of Bayesian optimization on two-dimensional test functions with contour
visualizations.

In addition to the previous functionality, this demo features
  - hyperparameter marginalization via SMC,
  - two-dimensional visualization via contour plots, and
  - visualization of the hyperparameter sampling (in Figure 2) via the pygp
    subpackage pygp.plotting.
"""

import numpy as np
import matplotlib.pyplot as pl
pl.rcParams['lines.linewidth'] = 2
pl.rcParams['image.cmap'] = 'coolwarm'

import pygp
import pygp.plotting as pp
import pybo


def animate2(model, bounds, info, x, index, ftrue):
    """
    Plot the current posterior and the index.
    """
    xx0, xx1 = np.meshgrid(                             # define grid
        np.linspace(bounds[0, 0], bounds[0, 1], 200),
        np.linspace(bounds[1, 0], bounds[1, 1], 200))
    xx = np.c_[xx0.flat, xx1.flat]

    ff = ftrue(xx).reshape(xx0.shape)                   # compute true function
    acq = index(xx).reshape(xx0.shape)                  # compute acquisition

    mu, _ = model.posterior(xx)                         # compute the posterior
    mu = mu.reshape(xx0.shape)

    X = info['x']                                       # get observations and
    xbest = info['xbest']                               # recommendations

    fig = pl.figure(1)
    pl.clf()

    pl.subplot(221)                                     # top left subplot
    pl.contour(xx0, xx1, ff, 20)                        # plot true function
    pl.scatter(x[0], x[1], color='k', zorder=2)         # plot current selection
    pl.scatter(X[:, 0], X[:, 1],                        # plot previous data
               facecolors='none', color='k', zorder=2)
    pl.scatter(xbest[-1, 0], xbest[-1, 1], marker='+',  # plot recommendation
               s=60, lw=2, color='g', zorder=2)
    pl.axis(bounds.flatten())
    pl.title('true function')

    pl.subplot(222)                                     # top right subplot
    pl.contour(xx0, xx1, mu, 20)                        # plot posterior
    pl.scatter(xbest[-1, 0], xbest[-1, 1], marker='+',  # plot recommendation
               s=60, lw=2, color='g', zorder=2)
    pl.axis(bounds.flatten())
    pl.title('posterior mean')

    pl.subplot(223)                                     # bottom left subplot
    pl.contour(xx0, xx1, acq, 20)                       # plot acquisition
    pl.scatter(x[0], x[1], color='k', zorder=2)         # plot current selection
    pl.axis(bounds.flatten())
    pl.title('acquisition')

    if not np.all(np.isnan(xbest)):
        pl.subplot(224)                                 # bottom right subplot
        pl.plot(ftrue(xbest), 'g', lw=2, alpha=0.5)     # plot value of recom.
        pl.axis('tight')
        pl.title('value of recommendation')

    for ax in fig.axes:                                 # remove tick labels
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    pl.draw()
    pl.show(block=False)

    pl.figure(2)                                        # plot hyperparameter
    pp.plot_samples(model)                              # samples

    pl.draw()
    pl.show(block=False)


if __name__ == '__main__':
    rng = 0                                             # random seed
    noise = 1e-5                                        # observation noise

    # define the objective function
    objective = pybo.functions.Branin(noise)

    bounds = objective.bounds                           # bounds of search space
    dim = bounds.shape[0]                               # dimension of space

    # prescribe initial hyperparameters
    sn = 1e-5                                           # likelihood std dev
    sf = 1.0                                            # kernel amplitude
    ell = 0.25 * (bounds[:, 1] - bounds[:, 0])          # kernel length scales
    mu = 0.0                                            # prior mean

    # define model
    kernel = 'matern5'                                  # kernel family
    gp = pygp.BasicGP(sn, sf, ell, mu, kernel=kernel)   # initialize base GP
    prior = {                                           # hyperparameter priors
        'sn': None,
        'sf': pygp.priors.LogNormal(np.log(sf), sigma=1., min=1e-6),
        'ell': pygp.priors.Uniform(ell / 100, ell * 2),
        'mu': pygp.priors.Gaussian(mu, sf)}
    model = pygp.meta.SMC(gp, prior, n=10, rng=rng)     # meta-model for SMC
                                                        # marginalization

    xrec, info, model = pybo.solve_bayesopt(
        objective,
        bounds,
        niter=30*dim,
        init='sobol',                                   # initialization policy
        policy='ei',                                    # exploration policy
        recommender='latent',                           # recommendation policy
        model=model,                                    # surrogate model
        rng=rng,
        callback=animate2)

    pl.show()
