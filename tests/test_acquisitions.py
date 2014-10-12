"""
Unit tests for different acquisition functions. This mainly tests that the
gradients of each acquisition function are computed correctly.
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
import pygp
import pybo.solvers.policies as policies


def check_acq_gradient(policy):
    # randomly generate some data.
    rng = np.random.RandomState(0)
    X = rng.rand(10, 2)
    y = rng.rand(10)

    # create the model.
    model = pygp.BasicGP(0.5, 1, [1, 1])
    model.add_data(X, y)

    # get the computed gradients.
    index = policy(model)
    xtest = rng.rand(20, 2)
    _, grad = index(xtest, grad=True)

    # numericall approximate the gradients
    index_ = lambda x: index(x[None])
    grad_ = np.array([spop.approx_fprime(x, index_, 1e-8) for x in xtest])

    nt.assert_allclose(grad, grad_, rtol=1e-6, atol=1e-6)


def test_acqs():
    for fname in policies.__all__:
        yield check_acq_gradient, getattr(policies, fname)
