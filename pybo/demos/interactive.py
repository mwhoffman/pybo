"""
This demo illustrates how to use pybo to optimize a black-box function that
requires a human in the loop. This script will prompt the user for a numerical
value at a particular design point every time it needs a new observation.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np

from ezplot import figure, show
from pybo import solve_bayesopt
from pybo.utils import InteractiveQuery

__all__ = []


def main():
    """Run the demo."""
    # initialize interactive function and 1d bounds
    f = InteractiveQuery()
    bounds = [0, 1]
    x = np.linspace(bounds[0], bounds[1], 100)

    # optimize the model and get final predictions
    xbest, model, info = solve_bayesopt(f, bounds, niter=10)
    mu, s2 = model.predict(x[:, None])

    # plot the final model
    fig = figure()
    axs = fig.gca()
    axs.plot_banded(x, mu, 2*np.sqrt(s2))
    axs.axvline(xbest)
    axs.scatter(info.x.ravel(), info.y)
    fig.canvas.draw()
    show()


if __name__ == '__main__':
    main()
