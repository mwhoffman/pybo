import numpy as np
import matplotlib.pyplot as pl

import pygp
import pybo.models
import pybo.policies


def run_model(model, policy, T):
    xmin = model.bounds[0,0]
    xmax = model.bounds[0,1]
    X = np.linspace(xmin, xmax, 200)[:, None]
    x = (xmax-xmin) / 2 + xmin

    pl.figure(1)
    pl.show()

    for i in xrange(T):
        y = model.get_data(x)
        policy.add_data(x, y)
        x = policy.get_next()

        pygp.gpplot(policy._gp, xmin=xmin, xmax=xmax, draw=False)

        pl.plot(X, model.f(X), lw=2, color='c')
        pl.plot(X, policy._index(X), lw=2)
        pl.axvline(x, color='r')
        pl.axis('tight')
        pl.axis(ymin=-3.4, ymax=3.4, xmin=xmin, xmax=xmax)
        pl.draw()


if __name__ == '__main__':
    sn = 0.2
    sf = 1.25
    ell = 0.05
    bounds = [0.5, 2.5]
    T = 100

    kernel = pygp.kernels.SEARD(sf, ell)
    model = pybo.models.GPModel(bounds, kernel, sn, rng=0)
    # model = pybo.models.Gramacy(sn)

    policy = pybo.policies.GPPolicy(model.bounds, sn, sf, ell, 'gppi')

    run_model(model, policy, T)
