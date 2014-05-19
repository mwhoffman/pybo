import numpy as np
import matplotlib.pyplot as pl

import pybo.models
import pygp


def run_optimization(model, policy, T):
    # grab the first point so we can initialize our datastructure.
    x = policy.get_next()

    # create the datastructure.
    d = len(x)
    data = np.zeros(T, [('x', np.float, (d,)),
                        ('y', np.float),
                        ('xstar', np.float, (d,)),
                        ('fstar', np.float),
                        ('regret', np.float)])

    for t in xrange(T):
        y = model.get_data(x)
        policy.add_data(x, y)
        data[t] = (x, y, policy.get_best(), np.nan, np.nan)
        x = policy.get_next() if (t < T-1) else None

    if hasattr(model, 'f'):
        data['fstar'] = model.f(data['xstar'])

    if hasattr(model, 'regret'):
        data['regret'] = model.regret(data['xstar'])

    return data


if __name__ == '__main__':
    # model parameters.
    sn = 0.2
    sf = 1.25
    ell = 0.05
    bounds = [0.5, 2.5]

    # horizon/repeats.
    T = 10
    N = 5

    data = dict()

    kernel = pygp.kernels.SEARD(sf, ell)
    model = pybo.models.GPModel(bounds, kernel, sn, rng=0)
    acqs = pybo.policies.gppolicy.ACQUISITION_FUNCTIONS.keys()

    for acq in acqs:
        data[acq] = []
        for n in xrange(N):
            print 'Running %s, %d' % (acq, n)
            policy = pybo.policies.GPPolicy(bounds, sn, sf, ell, acq)
            data[acq].append(run_optimization(model, policy, T))

    ax = pl.gca()
    ax.cla()
    for key in data.keys():
        dat = np.array(data[key])
        f = np.mean(dat['fstar'], axis=0)
        e = np.std(dat['fstar']) / np.sqrt(N)
        ax.plot(f, label=key, lw=2)
        ax.fill_between(range(T), f-e, f+e, alpha=0.1, color=ax.lines[-1].get_color())
        ax.legend(loc='best')
    ax.figure.canvas.draw()
