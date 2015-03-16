import numpy as np
import matplotlib.pyplot as pl

import benchfunk
import reggie

from pybo.bayesopt import inits
from pybo.bayesopt import policies
from pybo.bayesopt import solvers
from pybo.bayesopt import recommenders as recs


if __name__ == '__main__':
    f = benchfunk.Gramacy(0.2)
    X = inits.init_latin(f.bounds, 20)
    Y = f.get(X)

    xmin, xmax = f.bounds[0]
    Z = np.linspace(xmin, xmax, 500)[:, None]

    model = reggie.BasicGP(0.2, 1.9, 0.1, -1)
    model.add_data(X, Y)

    while True:
        xbest = recs.best_incumbent(model, f.bounds)
        index = policies.EI(model)
        xnext, _ = solvers.solve_lbfgs(index, f.bounds)
        T = Z.ravel()
        I = index(Z)

        pl.figure(1)
        pl.clf()
        color1 = next(pl.gca()._get_lines.color_cycle)
        color2 = next(pl.gca()._get_lines.color_cycle)

        pl.clf()
        pl.subplot(211)
        reggie.plotting.plot_posterior(model, xmin=xmin, xmax=xmax, draw=False)
        pl.axvline(xbest, ls='--', color=color1, label='recommendation')
        pl.axis(xmin=xmin, xmax=xmax)
        pl.legend(loc=0)

        pl.subplot(212)
        pl.plot(T, I, color=color2, label='acquisition')
        pl.fill_between(T, np.zeros_like(I), I, color=color2, alpha=0.1)
        pl.axvline(xnext, ls='--', color=color2, label='xnext')
        pl.axis(xmin=xmin, xmax=xmax)
        pl.legend(loc=0)
        pl.draw()

        model.add_data(xnext, f(xnext))
