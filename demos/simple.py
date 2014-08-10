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
        x, index = policy.get_next(return_index=True)

        # evaluate the index at the test points.
        index = np.log(index(X))
        index -= np.max(index)

        pl.clf()
        pl.plot(X, model.get_f(X), 'k--', lw=2, zorder=1)
        pl.plot(X, index, lw=2, zorder=2)
        pl.axvline(x, color='r', zorder=3)
        pl.axis('tight')
        pl.axis(ymin=-3.4, ymax=3.4, xmin=xmin, xmax=xmax)
        pl.draw()

        y = model(x)
        policy.add_data(x, y)


if __name__ == '__main__':
    T = 100
    sigma = 0.05
    model = pybo.models.Sinusoidal(sigma)
    policy = pybo.policies.GPPolicy(model.bounds, noise=sigma, policy='ei')

    run_model(model, policy, T)
