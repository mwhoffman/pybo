"""
Unit tests for different solver methods. This tests against the Branin-Hoo
function to make sure that all solvers are able to find the global optimum.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
import numpy as np
import numpy.testing as nt
import scipy.optimize as spop

# local imports
import pybo
import pybo.globalopt


def branin(x, grad=False):
    """
    Modified version of Branin-Hoo which computes its gradient if requested.
    """
    x = np.array(x, ndmin=2)
    n = x.shape[0]
    a = x[:,1] - (5.1 / (4 * np.pi**2)) * x[:,0]**2 + 5 * x[:,0] / np.pi - 6
    f = a**2 + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x[:,0]) + 10

    if not grad:
        return f

    da = np.vstack([5 / np.pi - 5.1 / (2 * np.pi**2) * x[:, 0], np.ones(n)])
    g = np.transpose(2 * a * da)
    g[:, 0] += -10 * (1 - 1 / (8 * np.pi)) * np.sin(x[:, 0])
    return f, g


# bounds for the branin function.
branin.bounds = np.array([[-5, 10], [0, 15]])
branin.fmin = 0.3978873


def test_branin():
    # get some test points
    rng = np.random.RandomState(0)
    width = branin.bounds[:, 1] - branin.bounds[:, 0]
    xtest = rng.rand(10, 2) * width + branin.bounds[:, 0]

    # get the gradient and a numerical approximation.
    df = branin(xtest, grad=True)[1]
    df_ = np.array([spop.approx_fprime(xk, branin, 1e-8) for xk in xtest])

    # define Branin-Hoo and its gradient
    nt.assert_allclose(df, df_, rtol=1e-6, atol=1e-6)


def check_solver_branin(solver):
    _, fmin = solver(branin, branin.bounds)
    nt.assert_allclose(fmin, branin.fmin, rtol=1e-6, atol=1e-6)


def test_global_solve_branin():
    for fname in pybo.globalopt.__all__:
        yield check_solver_branin, getattr(pybo.globalopt, fname)
