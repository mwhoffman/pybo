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
model = reggie.make_gp(1., 10., 0.1, 0.)
info, model = pybo.solve_bayesopt(prompter, bounds, model, niter=10)

# visualize
xx = np.linspace(0, 1, 100)
yy, _ = model.predict(xx[:,None])
X, y = model.data
plt.plot(xx, yy, lw=2)
plt.scatter(X, y, 50, lw=2, marker='o', facecolor='none')
plt.show()