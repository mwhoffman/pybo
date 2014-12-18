"""
Demo demonstrating Bayesian optimization with hyperparameter sampling. This
script includes the callback function `animate1` for user-defined visualization
and animation, which can be skipped upon first reading. Jump to '__main__' for
the meat of the script.

By default, pybo.solve_bayesopt() uses an MCMC meta-model to estimate
hyperparameter marginalization. In this tutorial we show how to define this
meta-model manually and prescribe our own hyperparameter priors. (Several priors
are implemented in pygp.priors.)

In this script we show how the user can define his own
  - kernel,
  - meta-model, or even
  - pass the GP model to solve_bayesopt() directly in order to fix the
    hyperparameters to their initially prescribed setting.
"""

import numpy as np
import matplotlib.pyplot as pl

import pygp
import pybo


def animate1(model, bounds, info, x, index, ftrue):
    """
    Plot the current posterior, the index, and the value of the current
    recommendation.
    """
    xmin, xmax = bounds[0]
    xx_ = np.linspace(xmin, xmax, 500)                  # define grid
    xx = xx_[:, None]

    ff = ftrue(xx)                                      # compute true function
    acq = index(xx)                                     # compute acquisition

    mu, s2 = model.posterior(xx)                        # compute posterior and
    lo = mu - 2 * np.sqrt(s2)                           # quantiles
    hi = mu + 2 * np.sqrt(s2)

    ymin, ymax = ff.min(), ff.max()                     # get plotting ranges
    ymin -= 0.2 * (ymax - ymin)
    ymax += 0.2 * (ymax - ymin)

    kwplot = {'lw': 2, 'alpha': 0.5}                    # common plotting kwargs

    fig = pl.figure(1)
    fig.clf()

    pl.subplot(221)
    pl.plot(xx, ff, 'k:', **kwplot)                     # plot true function
    pl.plot(xx, mu, 'b-', **kwplot)                     # plot the posterior and
    pl.fill_between(xx_, lo, hi, color='b', alpha=0.1)  # uncertainty bands
    pl.scatter(info['x'], info['y'],                    # plot data
               marker='o', facecolor='none', zorder=3)
    pl.axvline(x, color='r', **kwplot)                  # latest selection
    pl.axvline(info[-1]['xbest'], color='g', **kwplot)  # current recommendation
    pl.axis((xmin, xmax, ymin, ymax))
    pl.ylabel('posterior')

    pl.subplot(223)
    pl.fill_between(xx_, acq.min(), acq,                # plot acquisition
                    color='r', alpha=0.1)
    pl.axis('tight')
    pl.axvline(x, color='r', **kwplot)                  # plot latest selection
    pl.xlabel('input')
    pl.ylabel('acquisition')

    pl.subplot(222)
    pl.plot(ftrue(info['xbest']), 'g', **kwplot)        # plot performance
    pl.axis((0, len(info['xbest']), ymin, ymax))
    pl.xlabel('iterations')
    pl.ylabel('value of recommendation')

    for ax in fig.axes:                                 # remove tick labels
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    pl.draw()
    pl.show(block=False)


if __name__ == '__main__':
    rng = 0                                             # random seed
    noise = 1e-3                                        # observation noise

    # define the objective function
    objective = pybo.functions.Gramacy(noise)
    bounds = objective.bounds                           # bounds of search space
    dim = bounds.shape[0]                               # dimension of space

    # prescribe initial hyperparameters
    sn = noise                                          # likelihood std dev
    sf = 1.0                                            # kernel amplitude
    ell = 0.25 * (bounds[:, 1] - bounds[:, 0])          # kernel length scale
    mu = 0.0                                            # prior mean

    # define model
    kernel = 'matern3'                                  # kernel family
    gp = pygp.BasicGP(sn, sf, ell, mu, kernel=kernel)   # initialize base GP
    prior = {                                           # hyperparameter priors
        'sn': pygp.priors.Horseshoe(0.1, min=1e-5),
        'sf': pygp.priors.LogNormal(np.log(sf), sigma=1., min=1e-6),
        'ell': pygp.priors.Uniform(ell / 100, ell * 2),
        'mu': pygp.priors.Gaussian(mu, sf)}
    model = pygp.meta.MCMC(gp, prior, n=10, rng=rng)    # meta-model for MCMC
                                                        # marginalization

    xrec, info, model = pybo.solve_bayesopt(
        objective,
        bounds,
        niter=30*dim,
        init='sobol',                                   # initialization policy
        policy='ei',                                    # exploration policy
        recommender='incumbent',                        # recommendation policy
        model=model,                                    # surrogate model
        rng=rng,
        callback=animate1)

    pl.show()
