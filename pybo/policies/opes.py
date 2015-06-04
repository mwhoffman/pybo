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

from ..solvers import solve_lbfgs

__all__ = ['OPES']


def get_factors(m0, V0, fstar):
    """
    Given a Gaussian distribution with mean and covariance (m0, V0) use EP to
    find a Gaussian approximating the constraint that each latent variable is
    below fstar. Return the approximate factors (rho_, tau_) in canonical form.
    """
    # initialize the current state of the posterior as well as the EP factors
    # in canonical form.
    m, V = m0, V0
    rho_ = np.zeros_like(m0)
    tau_ = np.zeros_like(m0)

    # no damping on the first iteration
    damping = 1

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

        # update the EP factors with damping
        tau_ = tauNew * damping + tau_ * (1-damping)
        rho_ = rhoNew * damping + rho_ * (1-damping)

        # the new posterior.
        t = np.sqrt(tau_)
        L = linalg.cholesky(linalg.add_diagonal(t*V0*t[:, None], 1))
        V = linalg.solve_triangular(L, V0*t[:, None])
        V = V0 - np.dot(V.T, V)
        m = np.dot(V, rho_) + linalg.solve_cholesky(L, t*m0) / t

        if np.max(np.abs(np.r_[V.diagonal() - 1/tau, m - rho/tau])) >= 1e-6:
            damping *= 0.99
        else:
            break

    return rho_, tau_


class Predictor(object):
    def __init__(self, gp, fstar):
        # get the data and the noise variance
        X, Y = gp.data
        R = Y - gp._post.mean.get_function(X)
        sn2 = gp._post.like.get_variance()

        # get the mean and kernel at our latents
        m = gp._post.mean.get_function(X)
        K = gp._post.kern.get_kernel(X)

        # compute intermediate terms.
        L = linalg.cholesky(linalg.add_diagonal(K, sn2))
        A = linalg.solve_triangular(L, K)
        a = linalg.solve_triangular(L, R)

        # get the initial predictions
        m0 = m + np.dot(A.T, a)
        V0 = K - np.dot(A.T, A)

        # get the EP factors
        rho, tau = get_factors(m0, V0, fstar)
        omega = sn2 / (1 + sn2*tau)

        # get the cholesky of the covariance including the EP factors
        L = linalg.cholesky(linalg.add_diagonal(K, omega))

        # save the model
        self.gp = gp
        self.fstar = fstar

        # save everything
        self.rho = rho
        self.tau = tau
        self.omega = omega
        self.L = L

    def predict(self, Xtest):
        # get the data and the noise variance
        X, Y = self.gp.data
        R = Y - self.gp._post.mean.get_function(X)
        sn2 = self.gp._post.like.get_variance()

        # now evaluate the kernel at the new points and compute intermediate
        # terms
        K = self.gp._post.kern.get_kernel(X, Xtest)
        A = linalg.solve_triangular(self.L, K)
        a = linalg.solve_triangular(self.L, self.omega * (R/sn2 + self.rho))

        # get the predictions before the final constraint
        m1 = self.gp._post.mean.get_function(Xtest) + np.dot(A.T, a)
        v1 = self.gp._post.kern.get_dkernel(Xtest) - np.sum(A**2, axis=0)

        sigma = np.sqrt(v1)
        alpha = (self.fstar - m1) / sigma
        ratio = np.exp(ss.norm.logpdf(alpha) - ss.norm.logcdf(alpha))

        mu = m1 - ratio * sigma
        s2 = v1 - v1 * ratio * (ratio + alpha)

        return mu, s2


def OPES(model, bounds, nopt=50, nfeat=200, rng=None):
    rng = random.rstate(rng)

    if bounds.ndim == 2:
        funcs = [model.sample_f(nfeat, rng) for _ in xrange(nopt)]
        fopts = [solve_lbfgs(f.get, bounds, rng=rng)[1] for f in funcs]
    else:
        fopts = [model.sample(bounds).max() for _ in xrange(nopt)]

    preds = [Predictor(model, fopt) for fopt in fopts]

    def index(X, grad=False):
        if grad:
            raise NotImplementedError
        s2_pred = model.predict(X)[1] + model._post.like.get_variance()
        s2_cond = [pred.predict(X)[1] for pred in preds]
        s2_cond = np.array(s2_cond) + model._post.like.get_variance()
        H = 0.5 * (np.log(s2_pred) - np.sum(np.log(s2_cond), axis=0) / nopt)
        return H

    return index
