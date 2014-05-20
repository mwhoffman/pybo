import numpy as np
import matplotlib.pyplot as pl

import pybo.models
import pybo.policies


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
        y = model(x)
        policy.add_data(x, y)
        data[t] = (x, y, policy.get_best(), np.nan, np.nan)
        x = policy.get_next() if (t < T-1) else None

    if hasattr(model, 'get_f'):
        data['fstar'] = model.get_f(data['xstar'])

    if hasattr(model, 'get_regret'):
        data['regret'] = model.get_regret(data['xstar'])

    return data


if __name__ == '__main__':
    # model parameters.
    sn = 0.1
    sf = 1.25
    ell = 0.05

    # horizon/repeats.
    T = 50
    N = 20

    model = pybo.models.Gramacy(sn)
    bounds = model.bounds
    acqs = pybo.policies.gppolicy.ACQUISITION_FUNCTIONS.keys()

    data = dict()
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
        x = range(1,T+1)
        y = np.mean(dat['regret'], axis=0)
        e = np.std(dat['regret'], axis=0) / np.sqrt(N)
        ax.plot(x, y, label=key, lw=2)
        ax.fill_between(x, y+e, y-e, alpha=0.1, color=ax.lines[-1].get_color())
        ax.legend(loc='best')
    ax.set_title('Optimizer comparison on "%s"' % model.__class__.__name__)
    ax.set_yscale('log')
    ax.minorticks_off()
    ax.figure.canvas.draw()
