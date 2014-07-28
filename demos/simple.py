import numpy as np
import matplotlib.pyplot as pl

import pygp
import pygp.plotting
import pybo.models
import pybo.policies


def run_model(model, policy, T):
    xmin = model.bounds[0,0]
    xmax = model.bounds[0,1]
    X = np.linspace(xmin, xmax, 200)[:, None]

    x = (xmax-xmin) / 2 + xmin
    y = model(x)
    policy.add_data(x, y)

    pl.figure(1)
    pl.show()

    for i in xrange(T):
        x = policy.get_next()

        pl.clf()
        pl.plot(X, model.get_f(X), 'k--', lw=2, zorder=1)
        pl.plot(X, policy._index(X), lw=2, zorder=2)
        pl.scatter(policy._gp._X, policy._gp._y, color='m', zorder=2)
        pl.axvline(x, color='r', zorder=3)
        pl.axis('tight')
        pl.axis(ymin=-3.4, ymax=3.4, xmin=xmin, xmax=xmax)
        pl.draw()

        y = model(x)
        policy.add_data(x, y)


if __name__ == '__main__':
    # sn = 0.2
    # sf = 9.56
    # ell = 0.42
    # bounds = [0.5, 2.5]
    T = 100

    model = pybo.models.Sinusoidal(0.2)
    policy = pybo.policies.GPPolicy(model.bounds, policy='ei')

    run_model(model, policy, T)
