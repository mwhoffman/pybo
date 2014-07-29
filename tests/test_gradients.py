import numpy as np
import numpy.testing as nt
import scipy.optimize as spop

import pybo.models
import pybo.policies
from pybo.policies.gpacquisition import ei

def test_ei_gradients():
    rng = np.random.RandomState(0)
    model = pybo.models.Sinusoidal(0.2)
    policy = pybo.policies.GPPolicy(model.bounds, policy='ei')

    xmin = model.bounds[0, 0]
    xmax = model.bounds[0, 1]
    x = (xmax - xmin) / 2 + xmin
    for t in xrange(5):
        y = model(x)
        policy.add_data(x, y)
        x = policy.get_next()

    index = ei(policy._gp)

    # get some test points
    xtest = rng.rand(20) * (xmax - xmin) + xmin

    # compute gradients analytically
    _, dei = index(xtest[None])

    # define functions to feed into approx_fprime. These take a single point so
    # we have to use the None-index to "vectorize" it.
    fei = lambda x: index(x[None], grad=False)[0]
    # numerical approximation of gradients
    dei_ = np.array([spop.approx_fprime(xk, fei, 1e-8) for xk in xtest])

    # make sure everything is close to within tolerance
    nt.assert_allclose(dei, dei_, rtol=1e-6, atol=1e-6)


if __name__ == '__main__':
    test_ei_gradients()
