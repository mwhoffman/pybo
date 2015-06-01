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
    eta = np.zeros_like(m0)
    rho = np.zeros_like(m0)
    return eta, rho


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
    eta, rho = get_factors(m0, V0, fstar)
    omega = sn2 / (1 + sn2*rho)

    # get the cholesky of the kernel with omega on its diagonal
    L = linalg.cholesky(linalg.add_diagonal(K, omega))

    # now evaluate the kernel at the new points and compute intermediate terms
    K = kern.get_kernel(X, Xtest)
    A = linalg.solve_triangular(L, K)
    a = linalg.solve_triangular(L, omega * (R/sn2 + eta))

    # get the predictions
    mu = mean.get_function(Xtest) + np.dot(A.T, a)
    s2 = kern.get_dkernel(Xtest) - np.sum(A**2, axis=0)

    return mu, s2
