"""
Simple demo which performs Bayesian optimization on a single objective function
sampled from a known GP prior. This script also provides an example
visualization of the search process via the callback argument.
"""

import numpy as np
import matplotlib.pyplot as pl

import pygp
import pygp.plotting as pp
import pybo


def callback(model, bounds, info, x, index, ftrue):
    """
    Plot the current posterior, the index, and the value of the current
    recommendation.
    """
    xmin = bounds[0, 0]
    xmax = bounds[0, 1]

    xx = np.linspace(xmin, xmax, 500)[:, None]
    ff = ftrue(xx)
    ymin, ymax = ff.min(), ff.max()
    ymin -= 0.2 * (ymax - ymin)
    ymax += 0.2 * (ymax - ymin)

    # common plotting kwargs
    kwplot = {'lw': 2,
              'alpha': 0.5}

    fig = pl.figure(1)
    fig.clf()

    # plot the surrogate and data
    pl.subplot(221)
    pp.plot_posterior(model, xmin=xmin, xmax=xmax)
    pl.plot(xx, ff, 'k--', **kwplot)
    pl.axvline(x, color='r', **kwplot)
    pl.axvline(info[-1]['xbest'], color='g', **kwplot)
    pl.axis((xmin, xmax, ymin, ymax))
    pl.ylabel('posterior')

    # plot the acquisition function
    pl.subplot(223)
    pl.fill_between(xx.ravel(), 0., index(xx), color='r', alpha=0.1)
    pl.axis('tight')
    pl.axvline(x, color='r', **kwplot)
    pl.xlabel('input')
    pl.ylabel('acquisition')

    # plot the value at the current recommendation
    pl.subplot(222)
    pl.plot(ftrue(info['xbest']), 'g', **kwplot)
    pl.axis((0, len(info['xbest']), ymin, ymax))
    pl.xlabel('iterations')
    pl.ylabel('value of recommendation')

    for ax in fig.axes:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    pl.draw()
    pl.show(block=False)


if __name__ == '__main__':
    noise = 1e-6    # used as likelihood std dev as well as observation noise
    mean = 0.0      # GP prior mean (constant)
    rng = 0         # random seed

    # define GP prior from which the objective function will be sampled
    likelihood = pygp.likelihoods.Gaussian(noise)
    kernel = (pygp.kernels.Periodic(1, 1, 0.5) +
              pygp.kernels.SE(1, 1))
    gp = pygp.inference.ExactGP(likelihood, kernel, mean)

    # sample objective function from GP prior
    objective = pybo.functions.GPModel([3, 5], gp, rng=rng)

    dim = objective.bounds.shape[0]     # dimension of problem
    niter = 30 * dim                    # number of iterations (horizon)

    info = pybo.solve_bayesopt(objective,
                               objective.bounds,
                               T=niter,
                               policy='ei',
                               init='latin',
                               recommender='incumbent',
                               noisefree=True,
                               rng=rng,
                               callback=callback)

    pl.show()
