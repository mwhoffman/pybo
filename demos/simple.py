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

        pygp.plotting.plot(policy._model, xmin=xmin, xmax=xmax, subplot=211, draw=False)
        pl.plot(X, model.get_f(X), 'k--', lw=2)
        pl.axvline(x, color='r')
        pl.ylabel('posterior')
        pl.gca().set_xticklabels([])
        pl.gca().set_yticklabels([])

        pl.subplot(212)
        pl.cla()
        pl.plot(X, index(X), lw=2)
        pl.axvline(x, color='r')
        pl.axis(xmin=xmin, xmax=xmax)
        pl.ylabel('acquisition')
        pl.gca().set_xticklabels([])
        pl.gca().set_yticklabels([])
        pl.draw()

        y = model(x)
        policy.add_data(x, y)


if __name__ == '__main__':
    T = 100
    sigma = 0.05
    model = pybo.models.Sinusoidal(sigma)
    policy = pybo.policies.GPPolicy(model.bounds,
                                    noise=sigma,
                                    policy='ei',
                                    solver='lbfgs',
                                    inference='mcmc')

    run_model(model, policy, T)
