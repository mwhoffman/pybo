import numpy as np
import matplotlib.pyplot as pl

import pygp as pg
import pybo.models as pbm
import pybo.policies as pbp


if __name__ == '__main__':
    sn  = 0.2
    ell = 0.670104947766
    sf  = 1.25415619045

    model = pbm.Sinusoidal(0.2)
    gp = pg.BasicGP(sn, ell, sf)
    policy = pbp.GPUCB(gp, model.bounds)

    xmin = model.bounds[0][0]
    xmax = model.bounds[0][1]
    X = np.linspace(xmin, xmax, 200)[:, None]
    x = (xmax-xmin) / 2

    for i in xrange(40):
        pg.gpplot(policy.gp, xmin=xmin, xmax=xmax)
        pl.plot(X, policy.get_index(X), lw=2)
        pl.axvline(x, color='r')
        pl.axis('tight')
        pl.axis(xmin=xmin, xmax=xmax)
        pl.draw()

        y = model.get_data(x)
        policy.add_data(x, y)
        x = policy.get_next()
