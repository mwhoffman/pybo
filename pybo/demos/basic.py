"""
Animated demo showing progress of Bayesian optimization on a simple
(but highly multimodal) one-dimensional function.
"""

import matplotlib.pyplot as plt
import numpy as np

from reggie import make_gp, MCMC
from benchfunk.functions import Gramacy

from pybo import inits
from pybo import policies
from pybo import solvers
from pybo import recommenders


# we shouldn't be doing this. but I'll leave it for now.
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['legend.scatterpoints'] = 3
plt.rcParams['toolbar'] = 'None'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.color'] = 'k'
plt.rcParams['grid.linestyle'] = '-'
plt.rcParams['grid.alpha'] = 0.2
plt.rcParams['grid.linewidth'] = 0.5

if __name__ == '__main__':
    # grab a test function and points at which to plot things
    f = Gramacy(0.01)
    bounds = f.bounds

    # get initial data and some test points.
    X = inits.init_latin(bounds, 3)
    Y = np.array([f(x_) for x_ in X])
    x = np.linspace(bounds[0][0], bounds[0][1], 500)

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
    fig = plt.figure()
    fig.set_tight_layout(True)

    while True:
        # get index to solve it and plot it
        index = policies.EI(model, bounds, 0.1)

        # get the recommendation and the next query
        xbest = recommenders.best_latent(model, bounds)
        xnext, _ = solvers.solve_lbfgs(index, bounds)

        # evaluate the posterior and the acquisition function
        mu, s2 = model.predict(x[:, None])
        s = np.sqrt(s2)
        alpha = index(x[:, None])
        fbest += [f.get_f(xbest)]

        fig.clear()
        ax1 = plt.subplot2grid((2, 2), (0, 0))
        ax2 = plt.subplot2grid((2, 2), (1, 0), sharex=ax1)
        ax3 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)

        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax2.get_yticklabels(), visible=False)

        # plot the posterior and data
        ax1.set_title('current model')
        ax1.plot(x, mu, label='posterior')
        ax1.fill_between(x, mu - 2 * s, mu + 2 * s, alpha=0.1)
        ax1.plot(x, f.get_f(x[:, None]), label='true function')
        ax1.axvline(xbest, c='r', ls='--', label='recommendation')
        ax1.scatter(model.data[0].ravel(), model.data[1], label='data')

        # plot the acquisition function
        ax2.set_title('current policy')
        ax2.plot(x, alpha, label='acquisition')
        ax2.axvline(xnext, c='r', ls='--', label='next query')
        ax2.set_xlim(*bounds)

        # plot the latent function at recomended points
        ax3.set_title('value of recommendation')
        ax3.plot(fbest)

        # draw
        for ax in fig.axes:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.xaxis.set_tick_params(direction='out', top=False)
            ax.yaxis.set_tick_params(direction='out', right=False)
            ax.legend(loc=0)
        fig.canvas.draw()

        # add the next evaluation
        model.add_data(xnext, f(xnext))
