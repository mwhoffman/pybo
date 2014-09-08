"""
Wrapper class for simple GP-based policies whose acquisition functions are
simple functions of the posterior sufficient statistics.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
import numpy as np
import numpy.lib.recfunctions as rec

# not "exactly" local, but...
import pygp
import pygp.meta

# local imports
from .. import globalopt
from . import acquisitions

# exported symbols
__all__ = ['solve_bayesopt']


### ENUMERATE POSSIBLE META POLICY COMPONENTS #################################

def _make_dict(module, lstrip='', rstrip=''):
    """
    Given a module return a dictionary mapping the name of each of its exported
    functions to the function itself.
    """
    def generator():
        """Generate the (name, function) tuples."""
        for fname in module.__all__:
            f = getattr(module, fname)
            if fname.startswith(lstrip):
                fname = fname[len(lstrip):]
            if fname.endswith(rstrip):
                fname = fname[::-1][len(rstrip):][::-1]
            fname = fname.lower()
            yield fname, f
    return dict(generator())

MODELS = _make_dict(pygp.meta)
SOLVERS = _make_dict(globalopt, lstrip='solve_')
POLICIES = _make_dict(acquisitions)


#### DEFINE THE META POLICY ###################################################


def _get_best(model, bounds):
    """
    Given a model return the best recommendation, corresponding to the point
    with maximum posterior mean.
    """
    def mu(X, grad=False):
        if grad:
            return model.posterior(X, True)[::2]
        else:
            return model.posterior(X)[0]
    xinit, _ = model.data
    xbest, _ = globalopt.solve_lbfgs(mu, bounds, xx=xinit, maximize=True)
    return xbest


def solve_bayesopt(f,
                   bounds,
                   noise,
                   kernel,
                   solver='lbfgs',
                   policy='ei',
                   callback=None,
                   T=100):
    """
    Maximize the given function using Bayesian Optimization.
    """
    # make sure the bounds are a 2d-array.
    bounds = np.array(bounds, dtype=float, ndmin=2)
    d = len(bounds)

    # initialize the datastructure containing additional info.
    info = np.zeros(T, [('x', np.float, (d,)),
                        ('y', np.float),
                        ('xbest', np.float, (d,))])

    # initialize the policy components.
    model = pygp.inference.ExactGP(pygp.likelihoods.Gaussian(noise), kernel)
    solver = SOLVERS[solver]
    policy = POLICIES[policy]

    # create a list of initial points to query. For now just initialize with a
    # single point in the center of the bounds.
    init = [bounds.sum(axis=1) / 2.0]

    for i, x in enumerate(init):
        y = f(x)
        model.add_data(x, y)
        info[i] = (x, y, _get_best(model, bounds))

    for i in xrange(model.ndata, T):
        # get the next point to evaluate.
        index = policy(model)
        x, _ = solver(index, bounds, maximize=True)

        # deal with any visualization.
        if callback is not None:
            callback(info[:i], x, f, model, bounds, index)

        # make an observation and record it.
        y = f(x)
        model.add_data(x, y)
        info[i] = (x, y, _get_best(model, bounds))

    # if the function is an object with 'get_f' defined (for evaluating the
    # true function) then grab this data and record it.
    if hasattr(f, 'get_f'):
        info = rec.append_fields(info, 'fstar', f.get_f(info['xbest']),
                                 usemask=False)

    return info
