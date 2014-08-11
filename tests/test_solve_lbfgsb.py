# global imports
import numpy as np
import numpy.testing as nt
import scipy.optimize as spop

# local imports
import pybo
from pybo.policies._direct import solve_direct
from pybo.globalopt import solve_lbfgsb


def test_global_solve_branin():
    rng = np.random.RandomState(0)

    # define Branin-Hoo and its gradient
    bounds = np.array([[-5, 10], [0, 15]])
    def branin(x, grad=False):
        x = np.array(x, ndmin=2)
        n = x.shape[0]
        a = x[:,1] - (5.1 / (4 * np.pi**2)) * x[:,0]**2 + 5 * x[:,0] / np.pi - 6
        f = a**2 + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x[:,0]) + 10

        if grad:
            da = np.vstack([5 / np.pi - 5.1 / (2 * np.pi**2) * x[:, 0], np.ones(n)])
            g = np.transpose(2 * a * da)
            g[:, 0] += -10 * (1 - 1 / (8 * np.pi)) * np.sin(x[:, 0])

            return f, g

        return f

    # get some test points
    xtest = rng.rand(10, 2) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
    df = branin(xtest, grad=True)[1]
    # numerical approximation of gradient
    df_ = np.array([spop.approx_fprime(xk, branin, 1e-8) for xk in xtest])
    # make sure gradient is correct
    nt.assert_allclose(df, df_, rtol=1e-6, atol=1e-6)

    # run global optimizer
    xmin, fmin = solve_lbfgsb(branin, bounds)
    nt.assert_allclose(fmin, 0.3978873, rtol=1e-6, atol=1e-6)


def test_global_solve_direct():
    """Test global_solve against solve_direct."""

    rng = np.random.RandomState(0)
    model = pybo.models.Branin()
    policy_direct = pybo.policies.GPPolicy(model.bounds, policy='ei')
    policy_lbfgsb = pybo.policies.GPPolicy(model.bounds, policy='ei', solver='lbfgsb')

    xmin = model.bounds[:, 0]
    xmax = model.bounds[:, 1]
    x = (xmax - xmin) / 2 + xmin
    for t in xrange(4):
        y = model(x)
        policy_direct.add_data(x, y)
        policy_lbfgsb.add_data(x, y)
        x = policy_direct.get_next()

    # make sure everything is close to within tolerance
    nt.assert_allclose(x, policy_lbfgsb.get_next(), rtol=1e-6, atol=1e-6)

if __name__ == '__main__':
    test_global_solve_direct()