import numpy as np
import reggie
import mwhutils.random as random
import mwhutils.plotting as mp

import pybo.policies.pes


def visualize_pes(xmax):
    # seed the rng
    rng = random.rstate(0)

    # generate data from a GP prior.
    X = rng.rand(20, 1)
    Y = (reggie.BasicGP(sn2=0.1, rho=1, ell=0.05, kernel='matern3')
         .sample(X, latent=False, rng=rng))

    # create a new GP; note the different kernel
    gp = reggie.BasicGP(sn2=0.1, rho=1, ell=0.25)
    gp.add_data(X, Y)
    gp.optimize()

    # get the test locations.
    z = np.linspace(X.min(), X.max(), 200)

    # # get the "prior" predictions.
    mu0, s20 = gp.get_posterior(z[:, None])
    lo0 = mu0 - 2*np.sqrt(s20)
    hi0 = mu0 + 2*np.sqrt(s20)

    # get the "posterior" predictions.
    mu1, s21 = pybo.policies.pes.get_predictions(gp, xmax, z[:, None])
    lo1 = mu1 - 2*np.sqrt(s21)
    hi1 = mu1 + 2*np.sqrt(s21)

    fig = mp.figure(1)
    fig.scatter(X.ravel(), Y)
    fig.plot_banded(z, mu0, lo0, hi0)
    fig.plot_banded(z, mu1, lo1, hi1)
    fig.vline(xmax)
    fig.remove_ticks()
    fig.draw()


if __name__ == '__main__':
    visualize_pes(0.2)
