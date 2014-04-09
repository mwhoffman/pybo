import numpy as np
import matplotlib.pyplot as pl

import pybo.models as pbm
import pybo.utils.ldsample as pbs


if __name__ == '__main__':
    model = pbm.Sinusoidal(0.2)

    X = pbs.random(model.bounds, 100)
    y = np.hstack(model.get_data(x) for x in X)

    xmin = model.bounds[0][0]
    xmax = model.bounds[0][1]
    x = np.linspace(xmin, xmax, 200)
    f = -model.f(x)

    pl.gcf()
    pl.clf()
    pl.scatter(X, y)
    pl.plot(x, f, lw=2)
    pl.axis('tight')
    pl.draw()
