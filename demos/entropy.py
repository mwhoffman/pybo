import numpy as np
import matplotlib.pyplot as pl

import pygp
import pygp.plotting as pp

import pybo.solvers.policies.entropy as entropy


if __name__ == '__main__':
    gp = pygp.BasicGP(sn=0.1, ell=0.05, sf=1, kernel='matern3')

    rng = pygp.utils.random.rstate(0)
    X = rng.rand(20, 1)
    y = gp.sample(X, latent=False, rng=rng)

    gp = pygp.BasicGP(sn=0.1, ell=0.25, sf=1)
    gp.add_data(X, y)
    pygp.optimize(gp)

    pl.figure(1)
    pl.clf()
    pp.plot_posterior(gp)

    color = next(pl.gca()._get_lines.color_cycle)
    xx = np.linspace(X.min(), X.max(), 200)

    mu, Sigma = entropy.get_latent(gp, np.array([0.6]), xx[:, None])
    s2 = Sigma.diagonal()
    er = 2*np.sqrt(s2)

    pl.plot(xx, mu, color=color, lw=2)
    pl.fill_between(xx, mu+er, mu-er, color=color, alpha=0.15)

    pl.draw()
    pl.show()
