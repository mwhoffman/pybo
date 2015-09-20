"""
This demo illustrates how to use pybo to optimize a black-box function that
calls an external process. In particular this calls the command line calculator
`bc` to optimize a simple quadratic.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np

from ezplot import figure, show
from pybo import solve_bayesopt
from pybo.utils import SubprocessQuery

__all__ = []


def main():
    """Run the demo."""
    # grab a test function
    f = SubprocessQuery("bc <<< 'scale=8; x={}; -((x-3)^2)'")
    bounds = [0, 8]
    x = np.linspace(bounds[0], bounds[1], 500)

    # solve the model
    xbest, model, info = solve_bayesopt(f, bounds, niter=30, verbose=True)
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
