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


def get_latent(m0, v0, ymax, sn2):
    """
    Given a Gaussian (m0, v0) for the latent minimizer value return an
    approximate Gaussian posterior (m, v) subject to the constraint the value
    is greater than ymax, where the noise varaince sn2 is used to soften this
    constraint.
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


# NOTE: the following function is also implemented in pygp, and now might be a
# reasonable time to move some of these really common tasks to a third
# repository. Maybe.

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


def predict(gp, xstar, Xtest):
    """
    Given a GP posterior and a sampled location xstar return marginal
    predictions at Xtest conditioned on the fact that xstar is a minimizer.
    """
    kernel, sn2, mean, (X, y) = (gp._kernel, gp._likelihood.s2, gp._mean,
                                 gp.data)

    # format the optimum location as a (1,d) array.
    Z = xstar[None]

    # condition on our observations. NOTE: if this is an exact GP, then we've
    # already computed these quantities.
    Kxx = kernel.get(X) + sn2 * np.eye(X.shape[0])
    R = sla.cholesky(Kxx)
    a = sla.solve_triangular(R, y-mean, trans=True)

    # condition on the gradient being zero.
    Kxg = kernel.grady(X, Z)[:, 0, :]
    Kgg = kernel.gradxy(Z, Z)[0, 0]
    R, a = chol_update(R, Kxg, Kgg, a, np.zeros_like(xstar))

    # evaluate the kernel so we can test at the latent optimizer.
    Kzz = kernel.get(Z)
    Kzc = np.c_[
        kernel.get(Z, X),
        kernel.grady(Z, Z)[0]
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
    s2star = float(kernel.dget(Z) - np.sum(Bstar**2, axis=0))

    # evaluate the covariance between our test points and both the analytic
    # constraints and z.
    Ktc = np.c_[
        kernel.get(Xtest, X),
        kernel.grady(Xtest, Z)[:, 0],
        kernel.get(Xtest, Z)
    ]

    # get the marginal posterior without the constraint that the function at
    # the optimum is better than the function at test points.
    B = sla.solve_triangular(R, Ktc.T, trans=True)
    mu = mean + np.dot(B.T, a)
    s2 = kernel.dget(Xtest) - np.sum(B**2, axis=0)

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
