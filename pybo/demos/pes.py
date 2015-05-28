import numpy as np
import reggie
import mwhutils.random as random
import mwhutils.plotting as mp

import pybo.policies.pes


if __name__ == '__main__':
    # seed the rng
    rng = random.rstate(0)
    xmax = 0.2

    # generate data from a GP prior.
    X = rng.rand(20, 1)
    Y = (reggie.make_gp(sn2=0.1, rho=1, ell=0.05, kernel='matern3')
         .sample(X, latent=False, rng=rng))

    # create a new GP; note the different kernel
    gp = reggie.make_gp(sn2=0.1, rho=1, ell=0.25)
    gp.add_data(X, Y)
    gp.optimize()

    # get the test locations.
    z = np.linspace(X.min(), X.max(), 200)

    # get the PES index.
    bounds = np.array([[X.min(), X.max()]])
    index = pybo.policies.PES(gp, bounds, rng=rng)

    # # get the "prior" predictions.
    mu, s2 = gp.predict(z[:, None])

    # get the "posterior" predictions.
    mu_, s2_ = pybo.policies.pes.get_predictions(gp, xmax, z[:, None])

    fig = mp.figure(1, 2)
    fig[0].scatter(X.ravel(), Y)
    fig[0].plot_banded(z, mu, 2*np.sqrt(s2))
    fig[0].plot_banded(z, mu_, 2*np.sqrt(s2_))
    fig[0].vline(xmax)

    fig[1].plot_banded(z, index(z[:, None]))
    fig.draw()
