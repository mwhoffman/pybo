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
    # f = benchfunk.Sinusoidal(0.01)
    bounds = f.bounds

    # get initial data
    X = inits.init_latin(bounds, 10)
    Y = np.array([f(x_) for x_ in X])

    # initialize the model
    model = reggie.make_gp(0.01, 1.9, 0.1, 0)
    model.add_data(X, Y)

    model.params['like.sn2'].set_prior('uniform', 0.005, 0.015)

    model.params['kern.rho'].set_prior('lognormal', 0, 100)
    model.params['kern.ell'].set_prior('lognormal', 0, 10)
    model.params['mean.bias'].set_prior('normal', 0, 20)

    model = reggie.MCMC(model, n=20, rng=None)

    while True:
        xbest = recommenders.best_latent(model, bounds)
        index = policies.EI(model, bounds, 0.1)
        xnext, _ = solvers.solve_lbfgs(index, bounds)

        # get the posterior at test points
        x = np.linspace(bounds[0][0], bounds[0][1], 500)
        mu, s2 = model.predict(x[:, None])

        # create a figure and hold it
        fig = mp.figure(num=1, rows=2)
        fig.hold()

        # plot the posterior
        fig[0].plot_banded(x, mu, 2*np.sqrt(s2))
        fig[0].plot(x, f.get_f(x[:, None]))
        fig[0].scatter(model.data[0].ravel(), model.data[1])
        fig[0].vline(xbest)
        fig[0].vline(f.xopt)

        # plot the acquisition function
        fig[1].plot_banded(x, index(x[:, None]))
        fig[1].vline(xnext)

        # draw
        fig.draw()

        model.add_data(xnext, f(xnext))
