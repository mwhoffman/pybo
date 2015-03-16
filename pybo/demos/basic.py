import numpy as np
import matplotlib.pyplot as pl

import benchfunk
import reggie

import pybo.bayesopt.inits as inits


if __name__ == '__main__':
    f = benchfunk.Gramacy(0.2)
    X = inits.init_latin(f.bounds, 100)
    Y = f.get(X)

    model = reggie.BasicGP(0.2, 1.9, 0.1, -1)
    model.add_data(X, Y)

    pl.figure(1)
    reggie.plotting.plot_posterior(model)
