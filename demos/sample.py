import numpy as np
import matplotlib.pyplot as pl

import pygp
import pybo.models


if __name__ == '__main__':
    # sample a function from the GP prior and use that as a model. Note that
    # since we haven't added data, the noise term used by the GP model has no
    # affect whatsoever.
    gp = pygp.BasicGP(1, 0.65, 1)
    model = pybo.models.GPModel(gp, [0, 10], sigma=0.1, rng=0)

    X = np.linspace(model.bounds[0,0], model.bounds[0,1], 200)[:,None]

    pl.gcf()
    pl.cla()
    pl.plot(X, model.f(X))
    pl.draw()
