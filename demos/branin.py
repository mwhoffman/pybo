"""
Simple demo which performs Bayesian optimization on a single objective function
and visualizes the search process while doing so.
"""

import numpy as np
import matplotlib.pyplot as pl

import pygp
import pygp.plotting as pp
import pybo


def callback(info, x, f, model, _, index):
    """
    Plot the current posterior and the index.
    """
    z0 = np.linspace(f.bounds[0, 0], f.bounds[0, 1], 200)
    z1 = np.linspace(f.bounds[1, 0], f.bounds[1, 1], 200)
    Z0, Z1 = np.meshgrid(z0, z1)
    Z = np.c_[Z0.flat, Z1.flat]

    F = f.get_f(Z).reshape(Z0.shape)
    I = index(Z).reshape(Z0.shape)

    M, _ = model.posterior(Z)
    M = M.reshape(Z0.shape)

    X = info['x']
    xbest = info[-1]['xbest']

    pl.figure(1)
    pl.clf()

    pl.subplot(221)
    pl.contour(Z0, Z1, F, cmap='coolwarm')
    pl.axis(f.bounds.flatten())
    pl.gca().set_xticklabels([])
    pl.gca().set_yticklabels([])
    pl.title('true function')

    pl.scatter(x[0], x[1], marker='o', s=20, color='k', zorder=2)
    pl.scatter(X[:, 0], X[:, 1], facecolors='none', marker='o', s=20, lw=1,
               color='k', zorder=2)
    pl.scatter(xbest[0], xbest[1], marker='+', s=50, lw=2, color='k', zorder=2)

    pl.subplot(222)
    pl.contour(Z0, Z1, M, cmap='coolwarm')
    pl.axis(f.bounds.flatten())
    pl.gca().set_xticklabels([])
    pl.gca().set_yticklabels([])
    pl.title('posterior mean')
    pl.scatter(xbest[0], xbest[1], marker='+', s=50, lw=2, color='k', zorder=2)

    pl.subplot(223)
    pl.contour(Z0, Z1, I, cmap='coolwarm')
    pl.axis(f.bounds.flatten())
    pl.gca().set_xticklabels([])
    pl.gca().set_yticklabels([])
    pl.title('acquisition')

    pl.scatter(x[0], x[1], marker='o', s=20, color='k', zorder=2)

    if not all(np.isnan(info['fbest'])):
        pl.subplot(224)
        pl.semilogy(f.fmax - info['fbest'], lw=2)
        pl.axis('tight')
        pl.title('regret of recommendation')

    pl.draw()
    pl.show(block=False)

    pl.figure(2)
    pp.plot_samples(model)

    pl.draw()
    pl.show(block=False)


if __name__ == '__main__':
    T = 100
    sigma = 1e-10
    gp = pygp.BasicGP(sigma, 1.0, [0.1, 0.1], 0, kernel='matern3')
    f = pybo.functions.Branin(0.0)

    prior = {
        'sn': None,
        'sf': pygp.priors.Uniform(0.01, 5.0),
        'ell': pygp.priors.Uniform(np.full(2, 0.01), np.full(2, 5.0)),
        'mu': pygp.priors.Uniform(-10, 10)}

    info = pybo.solve_bayesopt(f,
                               f.bounds,
                               gp=gp,
                               prior=prior,
                               policy='ei',
                               recommender='incumbent',
                               inference='mcmc',
                               callback=callback)

    # this makes sure that if we run the demo from the command line that it
    # stops on the final plot before closing.
    pl.show()
