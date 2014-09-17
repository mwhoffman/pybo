"""
Predictive entropy search.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
import numpy as np
import scipy.linalg as sla
import scipy.stats as ss

# exported symbols
__all__ = []


def run_ep(m0, V0, ymin, sn2):
    # initial marginal approximation to our posterior given the zeroed factors
    # given below. note we're working with the "natural" parameters.
    tau = 1. / V0.diagonal()
    rho = m0 / V0.diagonal()

    # the current approximate factors.
    tauHat = np.zeros_like(tau)
    rhoHat = np.zeros_like(rho)

    # we won't do any damping at first.
    damping = 1

    while True:
        # eliminate the contribution of the approximate factor.
        v = (tau - tauHat) ** -1
        m = v * (rho - rhoHat)

        s = np.sqrt(v + sn2)
        t = ymin - m
        u = -1

        alpha = t / s
        ratio = np.exp(ss.norm.logpdf(alpha) - ss.norm.logcdf(alpha))
        beta = ratio * (alpha + ratio) / s / s
        kappa = u * (t / s + ratio) / s

        tauHatNew = beta / (1 - beta*v)
        tauHatNew[np.abs(tauHatNew) < 1e-300] = 1e-300
        rhoHatNew = (m + 1 / kappa) * tauHatNew

        # don't change anything that ends up with a negative variance.
        negv = (v < 0)
        tauHatNew[negv] = tauHat[negv]
        rhoHatNew[negv] = rhoHat[negv]

        while True:
            # mix between the new factors and the old ones. NOTE: in the first
            # iteration damping is 1, so this doesn't do any damping.
            tauHatNew = tauHatNew * damping + tauHat * (1-damping)
            rhoHatNew = rhoHatNew * damping + rhoHat * (1-damping)

            # get the eigenvalues of the new posterior covariance and mix more
            # with the old approximation if they're blowing up.
            vals, _ = np.linalg.eig(np.diag(1/tauHatNew) + V0)

            if any(1/vals <= 1e-10):
                damping *= 0.5
            else:
                break

        # our new approximate factors.
        tauHat = tauHatNew
        rhoHat = rhoHatNew

        # the new posterior.
        R = sla.cholesky(V0 + np.diag(1/tauHat))
        V = sla.solve_triangular(R, V0, trans=True)
        V = V0 - np.dot(V.T, V)
        m = np.dot(V, rhoHat) + sla.cho_solve((R, False), m0) / tauHat

        if np.max(np.abs(np.r_[V.diagonal() - 1/tau, m - rho/tau])) >= 1e-6:
            tau = 1 / V.diagonal()
            rho = m / V.diagonal()
            damping *= 0.99
        else:
            break

    vHat = 1 / tauHat
    mHat = rhoHat / tauHat

    return mHat, vHat


def predict(gp, xstar):
    X, y = gp.data

    # get the noise variance.
    sn2 = gp._likelihood.s2

    # construct the kernel matrix evaluated on our observed data c = [y; g]
    Ky = gp._kernel.get(X) + sn2 * np.eye(X.shape[0])
    Kyg = gp._kernel.grady(X, xstar[None])[:, 0, :]
    Kg = gp._kernel.gradxy(xstar[None], xstar[None])[0, 0]
    Kc = np.r_[np.c_[Ky, Kyg], np.c_[Kyg.T, Kg]]

    # construct the full kernel matrix. and get the sufficient statistics for
    # our posterior. Note that this ignores whatever inference method the GP is
    # using and performs exact inference.
    R = sla.cholesky(Kc)
    c = np.r_[y-gp._mean, np.zeros_like(xstar)]
    a = sla.solve_triangular(R, c, trans=True)

    # evaluate the kernel at our maximizer.
    Kzc = np.c_[
        gp._kernel.get(xstar[None], X),
        gp._kernel.grady(xstar[None], xstar[None])[0]]

    # get the mean and covariance of z given c.
    V = sla.solve_triangular(R, Kzc.T, trans=True)
    m0 = gp._mean + np.dot(V.T, a)
    V0 = gp._kernel.get(xstar[None]) - np.dot(V.T, V)

    mHat, vHat = run_ep(m0, V0, min(gp.data[1]), gp._likelihood.s2)

    return m0, V0, mHat, vHat
