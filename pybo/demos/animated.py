"""
Animated demo showing progress of Bayesian optimization on a simple
(but highly multimodal) one-dimensional function.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np

from reggie import make_gp, MCMC
from ezplot import figure, show

from pybo import inits
from pybo import policies
from pybo import solvers
from pybo import recommenders

__all__ = []


def f(x):
    """
    Test function we'll optimize. This is a 1d, sinusoidal function used by
    Gramacy and Lee in "Cases for the nugget in modeling computer experiments".
    """
    x = float(x)
    return -np.sin(10*np.pi*x) / (2*x) - (x-1)**4


def main():
    """Run the demo."""
    # define the bounds over which we'll optimize, the optimal x for comparison,
    # and a sequence of test points
    bounds = np.array([[0.5, 2.5]])
    xopt = 0.54856343
    x = np.linspace(bounds[0][0], bounds[0][1], 500)

    # get initial data and some test points.
    X = inits.init_latin(bounds, 3)
    Y = np.array([f(x_) for x_ in X])

    # initialize the model
    model = make_gp(0.01, 1.9, 0.1, 0)
    model.add_data(X, Y)

    # set a prior on the parameters
    model.params['like.sn2'].set_prior('uniform', 0.005, 0.015)
    model.params['kern.rho'].set_prior('lognormal', 0, 100)
    model.params['kern.ell'].set_prior('lognormal', 0, 10)
    model.params['mean.bias'].set_prior('normal', 0, 20)

    # make a model which samples parameters
    model = MCMC(model, n=20, rng=None)
    fbest = list()

    # create a new figure
    fig = figure(figsize=(10, 6))

    while True:
        # get index to solve it and plot it
        index = policies.EI(model, bounds, 0.1)

        # get the recommendation and the next query
        xbest = recommenders.best_observed(model, bounds)
        xnext, _ = solvers.solve_lbfgs(index, bounds)

        # evaluate the posterior and the acquisition function
        mu, s2 = model.predict(x[:, None])
        fbest.append(f(xbest))

        fig.clear()
        ax1 = fig.add_subplotspec((2, 2), (0, 0), hidex=True)
        ax2 = fig.add_subplotspec((2, 2), (1, 0), hidey=True, sharex=ax1)
        ax3 = fig.add_subplotspec((2, 2), (0, 1), rowspan=2)

        # plot the posterior and data
        ax1.plot_banded(x, mu, 2*np.sqrt(s2))
        ax1.scatter(model.data[0].ravel(), model.data[1])
        ax1.axvline(xbest)
        ax1.axvline(xnext, color='g')
        ax1.set_ylim(-6, 3)
        ax1.set_title('current model (xbest and xnext)')

        # plot the acquisition function
        ax2.plot_banded(x, index(x[:, None]))
        ax2.axvline(xnext, color='g')
        ax2.set_xlim(*bounds)
        ax2.set_title('current policy (xnext)')

        # plot the latent function at recomended points
        ax3.axhline(f(xopt))
        ax3.plot(fbest)
        ax3.set_ylim(0.4, 0.9)
        ax3.set_title('value of recommendation')

        # draw
        fig.canvas.draw()
        show()

        # add the next evaluation
        model.add_data(xnext, f(xnext))


if __name__ == '__main__':
    main()
