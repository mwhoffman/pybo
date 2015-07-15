"""
This demo illustrates how to use pybo to optimize a black-box function
that requires a human in the loop. This script will prompt the user
for a numerical value at a particular design point every time it
needs a new observation.
"""

import matplotlib.pyplot as plt
import numpy as np
import pybo
import reggie
from pybo import utils

# initialize prompter and 1d bounds
prompter = utils.Interactive()
bounds = np.array([0., 1.], ndmin=2)

# define model and optimize
model = reggie.make_gp(.1, 10., 0.1, 0.)
model = reggie.MCMC(model, n=10)
info, model = pybo.solve_bayesopt(prompter, bounds, model, niter=10)

# visualize
xx = np.linspace(0, 1, 100)
yy, s2 = model.predict(xx[:,None])
s = np.sqrt(s2)
X, y = model.data
plt.plot(xx, yy, lw=2)
plt.fill_between(xx, yy - 2 * s, yy + 2 * s, alpha=0.1)
plt.scatter(X, y, 50, lw=2, marker='o', facecolor='none')
plt.show()