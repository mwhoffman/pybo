import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import benchfunk
import reggie
from pybo import inits
from pybo import policies
from pybo import solvers
from pybo import recommenders


# makes plots prettier. should eventually be moved out of here.
mpl.rc('lines', lw=2)
mpl.rc('legend', scatterpoints=1)
mpl.rc('savefig', bbox='tight')

if __name__ == '__main__':
    # define noisy test function and search bounds
    sn2 = 0.1 ** 2                  # Gaussian noise variance
    f = benchfunk.Gramacy(sn2)
    bounds = f.bounds

    # get initial data and some test points.
    X = inits.init_latin(bounds, 3)
    Y = np.array([f(x_) for x_ in X])
    x = np.linspace(bounds[0][0], bounds[0][1], 500)

    # initialize the model
    rho = 2.0                       # initial kernel amplitude
    ell = 0.1                       # initial kernel length-scale
    mu = 0.0                        # constant prior mean
    model = reggie.make_gp(sn2, rho, ell, mu)
    model.add_data(X, Y)

    # set priors on the hyperparameters
    model.params['like.sn2'].set_prior('uniform', 0.005, 0.015)
    model.params['kern.rho'].set_prior('lognormal', 0, 100)
    model.params['kern.ell'].set_prior('lognormal', 0, 10)
    model.params['mean.bias'].set_prior('normal', 0, 20)

    # initialize a model with MCMC hyperparameter sampling
    model = reggie.MCMC(model, n=10)

    # create a new figure
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex='col', sharey='row')
    fig.show()
    axs[1,1].axis('off')
    fbest = list()

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

        # plot the posterior and data
        axs[0,0].clear()
        axs[0,0].plot(x, mu, label='posterior')
        axs[0,0].fill_between(x, mu - 2 * s, mu + 2 * s, alpha=0.1)
        axs[0,0].plot(x, f.get_f(x[:, None]), label='true function')
        axs[0,0].vlines(xbest, *axs[0, 0].get_ylim(), label='recommendation')
        axs[0,0].scatter(model.data[0].ravel(), model.data[1], label='data')

        # plot the acquisition function
        axs[1,0].clear()
        axs[1,0].plot(x, alpha, label='acquisition')
        axs[1,0].vlines(xnext, *axs[1, 0].get_ylim(), label='next query')
        axs[1,0].set_xlim(*bounds)

        # plot the latent function at recomended points
        axs[0,1].clear()
        fbest += [f.get_f(xbest)]
        axs[0,1].plot(fbest)
        axs[0,1].set_ylim([-7, 2])

        # draw
        [ax.legend(loc=0) for ax in axs.flatten()]
        fig.canvas.draw()

        # add the next evaluation
        model.add_data(xnext, f(xnext))
