"""
Predictive entropy search.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import scipy.stats as ss

import mwhutils.linalg as linalg
import mwhutils.random as random

__all__ = []


def get_factors(m0, V0, fstar):
    # initialize the current state of the posterior as well as the EP factors
    # in canonical form.
    m, V = m0, V0
    rho_ = np.zeros_like(m0)
    tau_ = np.zeros_like(m0)

    while True:
        # get the canonical form marginals
        tau = 1 / V.diagonal()
        rho = m / V.diagonal()

        # eliminate the contribution of the EP factors
        v = (tau - tau_) ** -1
        m = v * (rho - rho_)

        sigma = np.sqrt(v)
        alpha = (fstar - m) / sigma
        ratio = np.exp(ss.norm.logpdf(alpha) - ss.norm.logcdf(alpha))
        kappa = (ratio + alpha) / sigma
        gamma = ratio * kappa / sigma

        # get the new factors
        tauNew = gamma / (1 - gamma*v)
        rhoNew = (m - 1 / kappa) * tauNew

        # don't change anything that ends up with a negative variance.
        negv = (v < 0)
        tauNew[negv] = tau_[negv]
        rhoNew[negv] = rho_[negv]

        # update the EP factors
        tau_ = tauNew
        rho_ = rhoNew

        # the new posterior.
        t = np.sqrt(tau_)
        L = linalg.cholesky(linalg.add_diagonal(t*V0*t[:, None], 1))
        V = linalg.solve_triangular(L, V0*t[:, None])
        V = V0 - np.dot(V.T, V)
        m = np.dot(V, rho_) + linalg.solve_cholesky(L, t*m0) / t

        if np.max(np.abs(np.r_[V.diagonal() - 1/tau, m - rho/tau])) >= 1e-6:
            break

    return rho_, tau_


def get_predictions(gp, fstar, Xtest):
    # get the sub-models
    like = gp._post.like
    mean = gp._post.mean
    kern = gp._post.kern

    # get the data and the noise variance
    X, Y = gp.data
    R = Y - mean.get_function(X)
    sn2 = like.get_variance()

    # get the mean and kernel at our latents
    m = mean.get_function(X)
    K = kern.get_kernel(X)

    # compute intermediate terms.
    L = linalg.cholesky(linalg.add_diagonal(K, sn2))
    A = linalg.solve_triangular(L, K)
    a = linalg.solve_triangular(L, R)

    # get the initial predictions
    m0 = m + np.dot(A.T, a)
    V0 = K - np.dot(A.T, A)

    # get predictions of the latent variables at the observed inputs
    rho, tau = get_factors(m0, V0, fstar)
    omega = sn2 / (1 + sn2*tau)

    # get the cholesky of the kernel with omega on its diagonal
    L = linalg.cholesky(linalg.add_diagonal(K, omega))

    # now evaluate the kernel at the new points and compute intermediate terms
    K = kern.get_kernel(X, Xtest)
    A = linalg.solve_triangular(L, K)
    a = linalg.solve_triangular(L, omega * (R/sn2 + rho))

    # get the predictions
    mu = mean.get_function(Xtest) + np.dot(A.T, a)
    s2 = kern.get_dkernel(Xtest) - np.sum(A**2, axis=0)

    return mu, s2
