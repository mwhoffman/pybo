"""
Predictive entropy search.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import scipy.linalg as sla
import scipy.stats as ss

__all__ = []


def chol_update(A, B, C, a, b):
    """
    Update the cholesky decomposition of a growing matrix.
    Let `A` denote a cholesky decomposition of some matrix and `a` the inverse
    of `A` applied to some vector `y`. This computes the cholesky to a new
    matrix which has additional elements `B` and the non-diagonal and `C` on
    the diagonal block. It also computes the solution to the application of the
    inverse where the vector has additional elements `b`.
    """
    n = A.shape[0]
    m = C.shape[0]

    B = sla.solve_triangular(A, B, trans=True)
    C = sla.cholesky(C - np.dot(B.T, B))
    c = np.dot(B.T, a)

    # grow the new cholesky and use then use this to grow the vector a.
    A = np.r_[np.c_[A, B], np.c_[np.zeros((m, n)), C]]
    a = np.r_[a, sla.solve_triangular(C, b-c, trans=True)]

    return A, a


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
    sn2 = gp._sn2
    kernel = gp._kernel
    mean = gp._mean._bias
    X, y = gp.data

    # format the optimum location as a (1,d) array.
    Z = np.array(xstar, ndmin=2)

    # condition on our observations. NOTE: if this is an exact GP, then we've
    # already computed these quantities.
    Kxx = kernel.get_kernel(X) + sn2 * np.eye(X.shape[0])
    R = sla.cholesky(Kxx)
    a = sla.solve_triangular(R, y-mean, trans=True)

    # condition on the gradient being zero.
    Kgx = kernel.get_gradx(Z, X)[0]
    Kgg = kernel.get_gradxy(Z, Z)[0, 0]
    R, a = chol_update(R, Kgx, Kgg, a, np.zeros_like(xstar))

    # evaluate the kernel so we can test at the latent optimizer.
    Kzz = kernel.get_kernel(Z)
    Kzc = np.c_[
        kernel.get_kernel(Z, X),
        kernel.get_gradx(Z, Z)[0]
    ]

    # make predictions at the optimizer.
    B = sla.solve_triangular(R, Kzc.T, trans=True)
    m0 = mean + float(np.dot(B.T, a))
    v0 = float(Kzz - np.dot(B.T, B))

    # get the approximate factors and use this to update the cholesky, which
    # should now be wrt the covariance between [y; g; f(z)].
    m, v = get_latent(m0, v0, max(y), sn2)
    R, a = chol_update(R, Kzc.T, Kzz + v, a, m - mean)

    # get predictions at the optimum.
    Bstar = sla.solve_triangular(R, np.c_[Kzc, Kzz].T, trans=True)
    mustar = mean + float(np.dot(Bstar.T, a))
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
    B = sla.solve_triangular(R, Ktc.T, trans=True)
    mu = mean + np.dot(B.T, a)
    s2 = kernel.get_dkernel(Xtest) - np.sum(B**2, axis=0)

    # the covariance between each test point and xstar.
    rho = Ktc[:, -1] - np.dot(B.T, Bstar).flatten()
    s = s2 + s2star - 2*rho

    while any(s < 1e-10):
        rho[s < 1e-10] *= 1 - 1e-4
        s = s2 + s2star - 2*rho

    a = (mustar - mu) / np.sqrt(s)
    b = np.exp(ss.norm.logpdf(a) - ss.norm.logcdf(a))

    mu += b * (rho - s2) / np.sqrt(s)
    s2 -= b * (b + a) * (rho - s2)**2 / s

    return mu, s2
