"""
Demo which illustrates how to use solve_bayesopt as a simple method for global
optimization. The return values are the sequence of recommendations made by the
algorithm as well as the final model. The point `xbest[-1]` is the final
recommendation, i.e. the expected maximizer.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np

from ezplot import figure, show
from pybo import solve_bayesopt

__all__ = []


def f(x):
    """
    Test function that we will optimize. This is a simple sinusoidal function
    whose maximum should be found very quickly.
    """
    x = float(x)
    return -np.cos(x) - np.sin(3*x)


def main():
    """Run the demo."""
    # grab a test function
    bounds = [0, 2*np.pi]
    x = np.linspace(bounds[0], bounds[1], 500)

    # solve the model
    xbest, model, info = solve_bayesopt(f, bounds, niter=30, verbose=True)

    # make some predictions
    mu, s2 = model.predict(x[:, None])

    # plot the final model
    ax = figure().gca()
    ax.plot_banded(x, mu, 2*np.sqrt(s2))
    ax.axvline(xbest)
    ax.scatter(info.x.ravel(), info.y)
    ax.figure.canvas.draw()
    show()


if __name__ == '__main__':
    main()
