"""
Solver method for GP-based optimization which uses an inner-loop optimizer to
maximize some acquisition function, generally given as a simple function of the
posterior sufficient statistics.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
import numpy as np
import pygp

# each method/class defined exported by these modules will be exposed as a
# string to the solve_bayesopt method so that we can swap in/out different
# components for the "meta" solver.
from pygp import meta as models
from .. import globalopt
from . import policies

# exported symbols
__all__ = ['solve_bayesopt']


### HELPERS ###################################################################

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


def _get_best(model, bounds):
    """
    Given a model return the best recommendation, corresponding to the point
    with maximum posterior mean.
    """
    def mu(X, grad=False):
        """Posterior mean objective function."""
        if grad:
            return model.posterior(X, True)[::2]
        else:
            return model.posterior(X)[0]
    xinit, _ = model.data
    xbest, _ = globalopt.solve_lbfgs(mu, bounds, xx=xinit, maximize=True)
    return xbest


### SOLVER COMPONENTS #########################################################

MODELS = _make_dict(models)
SOLVERS = _make_dict(globalopt, lstrip='solve_')
POLICIES = _make_dict(policies)


### THE BAYESOPT META SOLVER ##################################################

def solve_bayesopt(f,
                   bounds,
                   solver='lbfgs',
                   policy='ei',
                   inference='mcmc',
                   gp=None,
                   prior=None,
                   T=100,
                   callback=None):
    """
    Maximize the given function using Bayesian Optimization.
    """
    # make sure the bounds are a 2d-array.
    bounds = np.array(bounds, dtype=float, ndmin=2)

    # grab the policy components (other than the model, which we'll initialize
    # after observing any initial data).
    solver = SOLVERS[solver]
    policy = POLICIES[policy]

    # create a list of initial points to query. For now just initialize with a
    # single point in the center of the bounds.
    X = [bounds.sum(axis=1) / 2.0]
    Y = [f(x) for x in X]

    if gp is None:
        # FIXME: the following default prior (and initial hyperparameter
        # setting) may not be the best in the world. But we can use X/Y to set
        # this if we want.
        sn, sf, mu = 1.0, 1.0, 0.0
        ell = (bounds[:, 1] - bounds[:, 0]) / 10
        gp = pygp.BasicGP(sn, sf, ell, mu, kernel='matern3')
        prior = {
            'sn': pygp.priors.Uniform(0.01, 1.0),
            'sf': pygp.priors.Uniform(0.01, 5.0),
            'ell': pygp.priors.Uniform(np.full_like(ell, 0.01), 2*ell),
            'mu': pygp.priors.Uniform(-10, 10)}

    if inference is 'fixed':
        model = gp.copy()

    elif prior is None:
        raise RuntimeError('cannot marginalize hyperparameters with no prior')

    else:
        # FIXME: this is assuming that the all inference methods correspond to
        # some Monte Carlo estimator with kwarg n.
        model = MODELS[inference](gp, prior, n=10)

    # add data to our model.
    model.add_data(X, Y)

    # allocate a datastructure containing "convergence" info.
    info = np.zeros(T, [('x', np.float, (len(bounds),)),
                        ('y', np.float),
                        ('xbest', np.float, (len(bounds),)),
                        ('fbest', np.float)])

    # initialize the data.
    info[:] = np.nan
    info['x'][:len(X)] = X
    info['y'][:len(Y)] = Y

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

        # find our next recommendation and evaluate it if possible.
        xbest = _get_best(model, bounds)
        fbest = f.get_f(xbest[None])[0] if hasattr(f, 'get_f') else np.nan

        # record everything.
        info[i] = (x, y, xbest, fbest)

    return info
