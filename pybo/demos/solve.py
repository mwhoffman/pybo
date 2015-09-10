"""
Demo which illustrates how to use solve_bayesopt as a simple method for global
optimization. The return values are the sequence of recommendations made by the
algorithm as well as the final model. The point `xbest[-1]` is the final
recommendation, i.e. the expected maximizer.
"""

import numpy as np

from ezplot import figure, show
from benchfunk import Sinusoidal
from pybo import solve_bayesopt


if __name__ == '__main__':
    # grab a test function
    f = Sinusoidal()

    # solve the model
    xbest, model = solve_bayesopt(f, f.bounds, niter=30, verbose=True)

    # make some predictions
    x = np.linspace(f.bounds[0, 0], f.bounds[0, 1], 500)
    mu, s2 = model.predict(x[:, None])

    # plot the final model
    fig = figure()
    axs = fig.gca()
    axs.plot_banded(x, mu, 2*np.sqrt(s2))
    axs.axvline(xbest[-1])
    axs.scatter(model.data[0].ravel(), model.data[1])
    fig.canvas.draw()
    show()
