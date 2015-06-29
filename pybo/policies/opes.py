"""
Predictive entropy search.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import scipy.stats as ss

import mwhutils.linalg as la
import mwhutils.random as random

from ..solvers import solve_lbfgs

__all__ = ['OPES']


def get_factors(m0, V0, fstar):
    """
    Given a Gaussian distribution with mean and covariance (m0, V0) use EP to
    find a Gaussian approximating the constraint that each latent variable is
    below fstar. Return the approximate factors (tau_, rho_) in canonical form.
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
        L = la.cholesky(la.add_diagonal(t*V0*t[:, None], 1))
        V = la.solve_triangular(L, V0*t[:, None])
        V = V0 - np.dot(V.T, V)
        m = np.dot(V, rho_) + la.solve_cholesky(L, t*m0) / t

        if np.max(np.abs(np.r_[V.diagonal() - 1/tau, m - rho/tau])) >= 1e-6:
            damping *= 0.99
        else:
            break

    return tau_, rho_


class Predictor(object):
    def __init__(self, gp, fstar):
        # get the data and the noise variance
        X, Y = gp.data
        R = Y - gp.mean.get_function(X)
        sn2 = gp.like.get_variance()

        # get the mean and kernel at our latents
        m = gp.mean.get_function(X)
        K = gp.kern.get_kernel(X)

        # compute intermediate terms.
        L = la.cholesky(la.add_diagonal(K, sn2))
        A = la.solve_triangular(L, K)
        a = la.solve_triangular(L, R)

        # get the initial predictions
        m0 = m + np.dot(A.T, a)
        V0 = K - np.dot(A.T, A)

        # get the EP factors and construct convolving factor
        tau, rho = get_factors(m0, V0, fstar)
        omega = sn2 / (1 + sn2*tau)

        # save the model
        self.gp = gp
        self.fstar = fstar

        # get the new posterior
        self.L = la.cholesky(la.add_diagonal(K, omega))
        self.a = la.solve_triangular(self.L, omega * (R/sn2 + rho))

    def predict(self, X):
        # now evaluate the kernel at the new points and compute intermediate
        # terms
        X_ = self.gp.data[0]
        K = self.gp.kern.get_kernel(X_, X)
        A = la.solve_triangular(self.L, K)

        # get the predictions before the final constraint
        m1 = self.gp.mean.get_function(X) + np.dot(A.T, self.a)
        v1 = self.gp.kern.get_dkernel(X) - np.sum(A**2, axis=0)

        # get the "prior" gradient at X
        dm1 = self.gp.mean.get_gradx(X)
        dv1 = self.gp.kern.get_dgradx(X)

        # get the kernel gradient and reshape it so we can do linear algebra
        dK = self.gp.kern.get_gradx(X, X_)
        dK = np.rollaxis(dK, 1)
        dK = np.reshape(dK, (dK.shape[0], -1))

        # compute the mean gradients
        dm1 += np.dot(dK.T, self.a).reshape(X.shape)

        # compute the variance gradients
        dA = la.solve_triangular(self.L, dK)
        dA = np.rollaxis(np.reshape(dA, (-1,) + X.shape), 2)
        dv1 -= 2 * np.sum(dA * A, axis=1).T

        sigma = np.sqrt(v1)
        alpha = (self.fstar - m1) / sigma
        ratio = np.exp(ss.norm.logpdf(alpha) - ss.norm.logcdf(alpha))
        kappa = ratio + alpha
        delta = ratio - alpha
        gamma = ratio * (delta*kappa + ratio*delta + 1)

        m2 = m1 - ratio * sigma
        v2 = v1 - v1 * ratio * (ratio + alpha)

        dm2 = (1 + ratio**2 - ratio * alpha)[:, None] * dm1
        dm2 += 0.5 * ((ratio*alpha*kappa - ratio**2) / sigma)[:, None] * dv1

        dv2 = (1 - ratio*kappa - 0.5*alpha*gamma)[:, None] * dv1
        dv2 -= (gamma*sigma)[:, None] * dm1

        return m2, v2, dm2, dv2


def OPES(model, bounds, nopt=50, nfeat=200, rng=None):
    rng = random.rstate(rng)
    fopts = [model.sample(bounds, rng=rng).max() for _ in xrange(nopt)]
    preds = [Predictor(model, fopt) for fopt in fopts]

    def index(X, grad=False):
        if grad:
            raise NotImplementedError
        s2_pred = model.predict(X)[1] + model.like.get_variance()
        s2_cond = [pred.predict(X)[1] for pred in preds]
        s2_cond = np.array(s2_cond) + model.like.get_variance()
        H = 0.5 * (np.log(s2_pred) - np.sum(np.log(s2_cond), axis=0) / nopt)
        return H

    return index
