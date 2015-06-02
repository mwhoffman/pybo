import numpy as np
import benchfunk
import reggie

import mwhutils.plotting as mp
import mwhutils.grid as mg

from pybo import inits
from pybo import policies
from pybo import solvers
from pybo import recommenders


if __name__ == '__main__':
    # grab a test function and points at which to plot things
    s = 0.5
    model = reggie.make_gp(s, 1.1, 0.02, 0)
    bounds = [[0, 5]]

    f = benchfunk.PriorFunction(model, bounds, 100, 2)
    x = mg.regular(bounds, 500)

    # get initial data
    X = inits.init_middle(bounds)
    Y = np.array([f(x_) for x_ in X])

    # initialize the model
    model.add_data(X, Y)

    while True:
        xbest = recommenders.best_latent(model, bounds)
        index = policies.OPES(model, bounds)(x)
        xnext = x[index.argmax()]

        # get the posterior at test points
        mu, s2 = model.predict(x)

        # create a figure and hold it
        fig = mp.figure(num=1, rows=2)
        fig.hold()

        # plot the posterior
        fig[0].plot_banded(x.ravel(), mu, 2*np.sqrt(s2))
        fig[0].plot(x.ravel(), f.get_f(x))
        fig[0].scatter(model.data[0].ravel(), model.data[1])
        fig[0].vline(xbest)

        # plot the acquisition function
        fig[1].plot_banded(x.ravel(), index)
        fig[1].vline(xnext)

        # draw
        fig.draw()

        model.add_data(xnext, f(xnext))
