"""
Beginner
========

Simplest demo performing Bayesian optimization on a one-dimensional test
function. This script also demonstrates user-defined visualization via a
callback function that is imported from the advanced demo.

The pybo.solve_bayesopt() function returns a numpy structured array, called
info below, which includes the observed input and output data, info['x'] and
info['y'], respectively; and the recommendations made along the way in
info['xbest'].

The callback function plots the posterior with uncertainty bands, overlayed onto
the true function; below it we plot the acquisition function, and to the right,
the evolution of the recommendation over time.
"""

import pybo

# import callback from advanced demo
import os
import sys
sys.path.append(os.path.dirname(__file__))
from advanced import callback


if __name__ == '__main__':
    objective = pybo.functions.Sinusoidal()

    info = pybo.solve_bayesopt(
        objective,
        objective.bounds,
        noisefree=True,
        rng=0,
        callback=callback)
