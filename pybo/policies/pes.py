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

from .thompson import FourierSample
from ..solvers import solve_lbfgs

__all__ = ['PES']


def get_latent(m0, v0, ymax, sn2):
    """
    Given a Gaussian (m0, v0) for the value of the latent maximizer return an
    approximate Gaussian posterior (m, v) subject to the constraint that the
    value is greater than ymax, where the noise varaince sn2 is used to soften
    this constraint.
    """
    s = np.sqrt(v0 + sn2)
    t = m0 - ymax

    alpha = t / s
    ratio = np.exp(ss.norm.logpdf(alpha) - ss.norm.logcdf(alpha))
    beta = ratio * (alpha + ratio) / s / s
    kappa = (alpha + ratio) / s

    m = m0 + 1. / kappa
    v = (1 - beta*v0) / beta

    return m, v


def get_predictions(gp, xstar, Xtest):
    """
    Given a GP posterior and a sampled location xstar return marginal
    predictions at Xtest conditioned on the fact that xstar is a minimizer.
    """
    sn2 = gp._post.like.get_variance()
    kernel = gp._post.kern
    mean = gp._post.mean

    # format the optimum location as a (1,d) array.
    X, Y = gp.data
    Z = np.array(xstar, ndmin=2)

    # condition on our observations. NOTE: if this is an exact GP, then we've
    # already computed these quantities.
    Kxx = linalg.add_diagonal(kernel.get_kernel(X), sn2)
    L = linalg.cholesky(Kxx)
    a = linalg.solve_triangular(L, Y - mean.get_function(X))

    # condition on the gradient being zero.
    Kgx = kernel.get_gradx(Z, X)[0]
    Kgg = kernel.get_gradxy(Z, Z)[0, 0]
    L, a = linalg.cholesky_update(L, Kgx.T, Kgg, a, np.zeros_like(xstar))

    # evaluate the kernel so we can test at the latent optimizer.
    Kzz = kernel.get_kernel(Z)
    Kzc = np.c_[
        kernel.get_kernel(Z, X),
        kernel.get_gradx(Z, Z)[0]
    ]

    # make predictions at the optimizer.
    B = linalg.solve_triangular(L, Kzc.T)
    m0 = float(np.dot(B.T, a)) + mean.get_function(Z)
    v0 = float(Kzz - np.dot(B.T, B))

    # get the approximate factors and use this to update the cholesky, which
    # should now be wrt the covariance between [y; g; f(z)].
    m, v = get_latent(m0, v0, max(Y), sn2)
    L, a = linalg.cholesky_update(L, Kzc, Kzz + v, a, m - mean.get_function(Z))

    # get predictions at the optimum.
    Bstar = linalg.solve_triangular(L, np.c_[Kzc, Kzz].T)
    mustar = float(np.dot(Bstar.T, a)) + mean.get_function(Z)
    s2star = float(kernel.get_dkernel(Z) - np.sum(Bstar**2, axis=0))

    # evaluate the covariance between our test points and both the analytic
    # constraints and z.
    Ktc = np.c_[
        kernel.get_kernel(Xtest, X),
        kernel.get_gradx(Z, Xtest)[0],
        kernel.get_kernel(Xtest, Z)
    ]

    # get the marginal posterior without the constraint that the function at
    # the optimum is better than the function at test points.
    B = linalg.solve_triangular(L, Ktc.T)
    mu = mean.get_function(Xtest) + np.dot(B.T, a)
    s2 = kernel.get_dkernel(Xtest) - np.sum(B**2, axis=0)

    # the covariance between each test point and xstar.
    rho = Ktc[:, -1] - np.dot(B.T, Bstar).flatten()
    s = s2 + s2star - 2*rho

    while any(s < 1e-10):
        rho[s < 1e-10] *= 1 - 1e-10
        s = s2 + s2star - 2*rho

    a = (mustar - mu) / np.sqrt(s)
    b = np.exp(ss.norm.logpdf(a) - ss.norm.logcdf(a))

    mu += b * (rho - s2) / np.sqrt(s)
    s2 -= b * (b + a) * (rho - s2)**2 / s

    return mu, s2


def PES(model, bounds, nopt=50, nfeat=200, rng=None):
    rng = random.rstate(rng)
    funcs = [FourierSample(model, nfeat, rng) for _ in xrange(nopt)]
    xopts = [solve_lbfgs(f.get, bounds, rng=rng)[0] for f in funcs]

    def index(X, grad=False):
        if grad:
            raise NotImplementedError
        s2_pred = model.predict(X)[1] + model._post.like.get_variance()
        s2_cond = [get_predictions(model, xopt, X)[1] for xopt in xopts]
        s2_cond = np.array(s2_cond) + model._post.like.get_variance()
        H = 0.5 * (np.log(s2_pred) - np.sum(np.log(s2_cond), axis=0) / nopt)
        return H

    return index
