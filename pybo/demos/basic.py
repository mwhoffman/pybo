import numpy as np
import matplotlib.pyplot as pl

import benchfunk
import reggie

from pybo.bayesopt import inits
from pybo.bayesopt import policies


if __name__ == '__main__':
    f = benchfunk.Gramacy(0.2)
    X = inits.init_latin(f.bounds, 100)
    Y = f.get(X)

    model = reggie.BasicGP(0.2, 1.9, 0.1, -1)
    model.add_data(X, Y)
    index_pi = policies.PI(model)

    Z = np.linspace(f.bounds[0][0], f.bounds[0][1], 500)[:, None]

    EI = policies.EI(model)(Z)
    EI += np.min(EI)
    EI /= np.max(EI)

    PI = policies.PI(model)(Z)
    PI += np.min(PI)
    PI /= np.max(PI)

    pl.figure(1)
    pl.clf()
    pl.subplot(211)
    reggie.plotting.plot_posterior(model)
    xmin, xmax, _, _ = pl.axis()

    pl.subplot(212)
    pl.plot(Z, EI, label='ei')
    pl.plot(Z, PI, label='pi')
    pl.legend(loc=0)
    pl.axis(xmin=xmin, xmax=xmax)
