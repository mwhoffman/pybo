import numpy as np
import numpy.testing as nt
import scipy.optimize as spop

import pybo.models
import pybo.policies
from pybo.policies.gpacquisition import ei

def test_gradients():
    rng = np.random.RandomState(0)
    model = pybo.models.Sinusoidal(0.2)
    policy_ei = pybo.policies.GPPolicy(model.bounds, policy='ei')
    policy_pi = pybo.policies.GPPolicy(model.bounds, policy='pi')
    policy_ucb = pybo.policies.GPPolicy(model.bounds, policy='ucb')

    xmin = model.bounds[0, 0]
    xmax = model.bounds[0, 1]
    x = (xmax - xmin) / 2 + xmin
    for t in xrange(5):
        y = model(x)
        policy_ei.add_data(x, y)
        policy_pi.add_data(x, y)
        policy_ucb.add_data(x, y)
        x = policy_ei.get_next()

    # get some test points
    xtest = rng.rand(20) * (xmax - xmin) + xmin
    xtest = xtest[None].T

    # compute gradients analytically
    _, dei = policy_ei._index(xtest, grad=True)
    _, dpi = policy_pi._index(xtest, grad=True)
    _, ducb = policy_ucb._index(xtest, grad=True)

    # define functions to feed into approx_fprime. These take a single point so
    # we have to use the None-index to "vectorize" it.
    fei = lambda x: policy_ei._index(x[None], grad=False)
    fpi = lambda x: policy_pi._index(x[None], grad=False)
    fucb = lambda x: policy_ucb._index(x[None], grad=False)

    # numerical approximation of gradients
    dei_ = np.array([spop.approx_fprime(xk, fei, 1e-8) for xk in xtest])
    dpi_ = np.array([spop.approx_fprime(xk, fpi, 1e-8) for xk in xtest])
    ducb_ = np.array([spop.approx_fprime(xk, fucb, 1e-8) for xk in xtest])

    # make sure everything is close to within tolerance
    nt.assert_allclose(dei, dei_, rtol=1e-6, atol=1e-6)
    nt.assert_allclose(dpi, dpi_, rtol=1e-6, atol=1e-6)
    nt.assert_allclose(ducb, ducb_, rtol=1e-6, atol=1e-6)


def test_gradients_2d():
    rng = np.random.RandomState(0)
    model = pybo.models.Branin()
    policy_ei = pybo.policies.GPPolicy(model.bounds, policy='ei')
    policy_pi = pybo.policies.GPPolicy(model.bounds, policy='pi')
    policy_ucb = pybo.policies.GPPolicy(model.bounds, policy='ucb')

    xmin = model.bounds[:, 0]
    xmax = model.bounds[:, 1]
    x = (xmax - xmin) / 2 + xmin
    for t in xrange(5):
        y = model(x)
        policy_ei.add_data(x, y)
        policy_pi.add_data(x, y)
        policy_ucb.add_data(x, y)
        x = policy_ei.get_next()

    # get some test points
    xtest = rng.rand(20, 2) * (xmax - xmin) + xmin

    # compute gradients analytically
    _, dei = policy_ei._index(xtest, grad=True)
    _, dpi = policy_pi._index(xtest, grad=True)
    _, ducb = policy_ucb._index(xtest, grad=True)

    # define functions to feed into approx_fprime. These take a single point so
    # we have to use the None-index to "vectorize" it.
    fei = lambda x: policy_ei._index(x[None], grad=False)
    fpi = lambda x: policy_pi._index(x[None], grad=False)
    fucb = lambda x: policy_ucb._index(x[None], grad=False)

    # numerical approximation of gradients
    dei_ = np.array([spop.approx_fprime(xk, fei, 1e-8) for xk in xtest])
    dpi_ = np.array([spop.approx_fprime(xk, fpi, 1e-8) for xk in xtest])
    ducb_ = np.array([spop.approx_fprime(xk, fucb, 1e-8) for xk in xtest])

    # make sure everything is close to within tolerance
    nt.assert_allclose(dei, dei_, rtol=1e-6, atol=1e-6)
    nt.assert_allclose(dpi, dpi_, rtol=1e-6, atol=1e-6)
    nt.assert_allclose(ducb, ducb_, rtol=1e-6, atol=1e-6)
