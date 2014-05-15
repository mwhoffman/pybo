import numpy as np
import matplotlib.pyplot as pl

import pygp as pg
import pybo.models as pbm
import pybo.policies as pbp


def run_model(Policy, Model, sn, sf, ell, T):
    model = Model(sn)
    gp = pg.BasicGP(sn, sf, ell)
    policy = Policy(gp, model.bounds)

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

        pg.gpplot(policy.gp, xmin=xmin, xmax=xmax, draw=False)

        pl.plot(X, policy.get_index(X), lw=2)
        pl.axvline(x, color='r')
        pl.axis('tight')
        pl.axis(ymin=-2.4, ymax=2.4, xmin=xmin, xmax=xmax)
        pl.draw()


def run_policy(policy):
    run_model(pbp.GPUCB, pbm.Gramacy, 0.2, 1.25, 0.05, 100)


if __name__ == '__main__':
    run_policy('gpucb')

