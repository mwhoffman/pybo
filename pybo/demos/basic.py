import numpy as np
import matplotlib.pyplot as pl

import benchfunk
import reggie

from pybo.bayesopt import inits
from pybo.bayesopt import policies
from pybo.bayesopt import solvers


if __name__ == '__main__':
    f = benchfunk.Gramacy(0.2)
    X = inits.init_latin(f.bounds, 10)
    Y = f.get(X)

    xmin, xmax = f.bounds[0]
    Z = np.linspace(xmin, xmax, 500)[:, None]

    model = reggie.BasicGP(0.2, 1.9, 0.1, -1)
    model.add_data(X, Y)

    pl.figure(1)
    pl.clf()
    pl.subplot(211)
    reggie.plotting.plot_posterior(model, xmin=xmin, xmax=xmax)
    pl.legend(loc=0)
    xmin, xmax, _, _ = pl.axis()

    pl.subplot(212)
    for policy in [policies.EI, policies.PI]:
        index = policy(model)
        xstar, _ = solvers.solve_lbfgs(index, f.bounds)
        vals = index(Z)
        vals += np.min(vals)
        vals /= np.max(vals)
        lines = pl.plot(Z, vals, label=policy.__name__.lower())
        pl.axvline(xstar, ls='--', color=lines[0].get_color())

    pl.legend(loc=0)
    pl.axis(xmin=xmin, xmax=xmax)
