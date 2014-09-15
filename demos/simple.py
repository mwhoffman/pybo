"""
Simple demo which performs Bayesian optimization on a single objective function
and visualizes the search process while doing so.
"""

import numpy as np
import matplotlib.pyplot as pl

import pygp
import pygp.plotting as pp
import pybo


def callback(info, x, f, model, bounds, index):
    """
    Plot the current posterior and the index.
    """
    xmin = bounds[0, 0]
    xmax = bounds[0, 1]

    X = np.linspace(xmin, xmax, 500)[:, None]
    F = f.get_f(X)
    ymin, ymax = F.min(), F.max()
    ymin -= 0.2 * (ymax - ymin)
    ymax += 0.2 * (ymax - ymin)

    pl.figure(1)
    pl.clf()
    pl.subplot(221)
    pp.plot_posterior(model, xmin=xmin, xmax=xmax)
    pl.plot(X, F, 'k--', lw=2)
    pl.axvline(x, color='r')
    pl.axvline(info[-1]['xbest'], color='g')
    pl.axis((xmin, xmax, ymin, ymax))
    pl.gca().set_xticklabels([])
    pl.gca().set_yticklabels([])
    pl.ylabel('posterior')

    pl.subplot(223)
    pl.plot(X, index(X), lw=2)
    pl.axvline(x, color='r')
    pl.axis(xmin=xmin, xmax=xmax)
    pl.gca().set_xticklabels([])
    pl.gca().set_yticklabels([])
    pl.xlabel('input')
    pl.ylabel('acquisition')

    pl.subplot(122)
    pl.plot(info['fbest'], lw=2)
    pl.axis('tight')
    pl.xlabel('iterations')
    pl.ylabel('value of recommendation')
    pl.gca().set_xticklabels([])
    pl.gca().set_yticklabels([])
    pl.draw()
    pl.show(block=False)


if __name__ == '__main__':
    T = 100
    sigma = 0.05
    gp = pygp.BasicGP(sigma, 1.0, 0.1, kernel='matern3')
    f = pybo.functions.GPModel([3, 5], gp)

    info = pybo.solve_bayesopt(f,
                               f.bounds,
                               policy='ei',
                               recommender='incumbent',
                               inference='mcmc',
                               callback=callback)

    # this makes sure that if we run the demo from the command line that it
    # stops on the final plot before closing.
    pl.show()
