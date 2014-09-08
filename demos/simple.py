"""
Simple demo which performs Bayesian optimization on a single objective function
and visualizes the search process while doing so.
"""

import numpy as np
import matplotlib.pyplot as pl

import pygp
import pygp.plotting as pp
import pybo.models
import pybo.policies


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
    ymax += 0.4 * (ymax - ymin)

    pl.subplot(211)
    pl.cla()
    pp.plot_posterior(model, xmin=xmin, xmax=xmax)
    pl.axis((xmin, xmax, ymin, ymax))

    pl.plot(X, F, 'k--', lw=2)
    pl.axvline(x, color='r')
    pl.axvline(info[-1]['xbest'], color='g')
    pl.gca().set_xticklabels([])
    pl.gca().set_yticklabels([])
    pl.ylabel('posterior')

    pl.subplot(212)
    pl.cla()
    pl.plot(X, index(X), lw=2)
    pl.axvline(x, color='r')
    pl.axis(xmin=xmin, xmax=xmax)
    pl.gca().set_xticklabels([])
    pl.gca().set_yticklabels([])
    pl.ylabel('acquisition')
    pl.draw()


if __name__ == '__main__':
    T = 100
    sigma = 0.05
    gp = pygp.BasicGP(sigma, 1.0, 0.1, kernel='matern3')
    f = pybo.models.GPModel([3, 5], gp)

    pybo.policies.solve_bayesopt(f,
                                 f.bounds,
                                 sigma,
                                 policy='ucb',
                                 kernel=gp._kernel.copy(),
                                 callback=callback)
