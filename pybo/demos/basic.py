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
    f = benchfunk.Gramacy(0.2)
    x = np.linspace(f.bounds[0, 0], f.bounds[0, 1], 500)

    # get initial data
    X = inits.init_latin(f.bounds, 20)
    Y = f.get(X)

    # initialize the model
    model = reggie.BasicGP(0.2, 1.9, 0.1, -1)
    model.add_data(X, Y)

    while True:
        xbest = recommenders.best_incumbent(model, f.bounds)
        index = policies.EI(model)
        xnext, _ = solvers.solve_lbfgs(index, f.bounds)

        mu, s2 = model.get_posterior(x[:, None])
        lo = mu - 2*np.sqrt(s2)
        hi = mu + 2*np.sqrt(s2)

        # plot the observed data
        fig = mp.figure(1, 3)
        fig[0].scatter(model.data[0].ravel(), model.data[1])

        # plot the posterior
        fig[1].plot_banded(x, mu, lo, hi)
        fig[1].plot(x, f.get_f(x[:, None]), ls='--', color='k')
        fig[1].vline(xbest)

        # plot the acquisition function
        fig[2].plot_banded(x, index(x[:, None]))
        fig[2].vline(xnext)

        # draw
        fig.draw()

        model.add_data(xnext, f(xnext))
