import numpy as np
import benchfunk
import reggie
import mwhutils.plotting as mp

from pybo import inits
from pybo import policies
from pybo import solvers
from pybo import recommenders


if __name__ == '__main__':
    # grab a test function and points at which to plot things
    f = benchfunk.Gramacy(0.01)
    bounds = f.bounds

    # get initial data and some test points.
    X = inits.init_latin(bounds, 3)
    Y = np.array([f(x_) for x_ in X])
    x = np.linspace(bounds[0][0], bounds[0][1], 500)

    # initialize the model
    model = reggie.make_gp(0.01, 1.9, 0.1, 0)
    model.add_data(X, Y)

    # set a prior on the parameters
    model.params['like.sn2'].set_prior('uniform', 0.005, 0.015)
    model.params['kern.rho'].set_prior('lognormal', 0, 100)
    model.params['kern.ell'].set_prior('lognormal', 0, 10)
    model.params['mean.bias'].set_prior('normal', 0, 20)

    # make a model which samples parameters
    model = reggie.MCMC(model, n=20, rng=None)

    # create a new figure
    fig = mp.figure(rows=2, figsize=(6, 8))

    while True:
        # get the index so we can both solve it and plot it
        index = policies.EI(model, bounds, 0.1)

        # get the recommendation and the next query
        xbest = recommenders.best_latent(model, bounds)
        xnext, _ = solvers.solve_lbfgs(index, bounds)

        # evaluate the posterior and the acquisition function
        mu, s2 = model.predict(x[:, None])
        alpha = index(x[:, None])

        # clear the figure
        fig.clear()
        fig.hold()

        # plot the posterior
        fig[0].plot(x, mu, 2*np.sqrt(s2), label='posterior')
        fig[0].plot(x, f.get_f(x[:, None]), label='true function')
        fig[0].scatter(model.data[0].ravel(), model.data[1], label='data')
        fig[0].vline(xbest, label='recommendation')
        fig[0].set_lim(ymin=-7, ymax=2)

        # plot the acquisition function
        fig[1].plot(x, alpha, 0, add=False, label='acquisition')
        fig[1].vline(xnext, label='next query')
        fig[1].remove_ticks(yticks=True)

        # draw
        fig.draw()

        # add the next evaluation
        model.add_data(xnext, f(xnext))
