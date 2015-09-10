"""
This demo illustrates how to use pybo to optimize a black-box function
that requires a human in the loop. This script will prompt the user
for a numerical value at a particular design point every time it
needs a new observation.
"""

import numpy as np

from ezplot import figure, show
from benchfunk import Interactive
from pybo import solve_bayesopt


if __name__ == '__main__':
    # initialize prompter and 1d bounds
    prompter = Interactive()
    bounds = np.array([0., 1.], ndmin=2)

    # define model and optimize
    xbest, model = solve_bayesopt(prompter, bounds, niter=10)

    # get our predictions
    x = np.linspace(0, 1, 100)
    mu, s2 = model.predict(x[:, None])

    # plot the final model
    fig = figure()
    axs = fig.gca()
    axs.plot_banded(x, mu, 2*np.sqrt(s2))
    axs.axvline(xbest[-1])
    axs.scatter(model.data[0].ravel(), model.data[1])
    fig.canvas.draw()
    show()
