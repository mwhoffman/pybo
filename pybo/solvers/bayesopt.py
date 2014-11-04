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

from numpy.lib.recfunctions import append_fields

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


### SOLVER COMPONENTS #########################################################

# each method/class defined exported by these modules will be exposed as a
# string to the solve_bayesopt method so that we can swap in/out different
# components for the "meta" solver.
from .. import globalopt as solvers
from ..utils.random import rstate
from . import init as initializers
from . import policies
from . import recommenders

POLICIES = _make_dict(policies)
INITIALIZERS = _make_dict(initializers, lstrip='init_')
SOLVERS = _make_dict(solvers, lstrip='solve_')
RECOMMENDERS = _make_dict(recommenders, lstrip='best_')


### THE BAYESOPT META SOLVER ##################################################

def solve_bayesopt(f,
                   bounds,
                   T=100,
                   policy='ei',
                   init='middle',
                   solver='lbfgs',
                   recommender='latent',
                   model=None,
                   noisefree=False,
                   ftrue=None,
                   rng=None,
                   callback=None):
    """
    Maximize the given function using Bayesian Optimization.
    """
    rng = rstate(rng)

    # make sure the bounds are a 2d-array.
    bounds = np.array(bounds, dtype=float, ndmin=2)

    # see if the query object itself defines ground truth.
    if (ftrue is None) and hasattr(f, 'get_f'):
        ftrue = f.get_f

    # initialize all the solver components.
    policy = POLICIES[policy]
    init = INITIALIZERS[init]
    solver = SOLVERS[solver]
    recommender = RECOMMENDERS[recommender]

    # create a list of initial points to query.
    X = init(bounds, rng)
    Y = [f(x) for x in X]

    if model is None:
        # initialize parameters of a simple GP model.
        sf = np.std(Y) if (len(Y) > 1) else 10.
        mu = np.mean(Y)
        ell = bounds[:, 1] - bounds[:, 0]

        # FIXME: this may not be a great setting for the noise parameter; if
        # we're noisy it may not be so bad... but for "noisefree" models this
        # sets it fixed to 1e-6, which may be too big.
        sn = 1e-6 if noisefree else 1e-3

        # specify a hyperprior for the GP.
        prior = {
            'sn': (
                None if noisefree else
                pygp.priors.Horseshoe(scale=0.1, min=1e-6)),
            'sf': pygp.priors.LogNormal(mu=np.log(sf), sigma=1., min=1e-6),
            'ell': pygp.priors.Uniform(ell / 100, ell * 2),
            'mu': pygp.priors.Gaussian(mu, sf)}

        # create the GP model (with hyperprior).
        model = pygp.BasicGP(sn, sf, ell, mu, kernel='matern5')
        model = pygp.meta.MCMC(model, prior, n=10, burn=100, rng=rng)

    # add any initial data to our model.
    model.add_data(X, Y)

    # allocate a datastructure containing "convergence" info.
    info = np.zeros(T, [('x', np.float, (len(bounds),)),
                        ('y', np.float),
                        ('xbest', np.float, (len(bounds),))])

    # initialize the data.
    info['x'][:len(X)] = X
    info['y'][:len(Y)] = Y

    for i in xrange(model.ndata, T):
        # get the next point to evaluate.
        index = policy(model)
        x, _ = solver(index, bounds, maximize=True, rng=rng)

        # deal with any visualization.
        if callback is not None:
            callback(model, bounds, info[:i], x, index, ftrue)

        # make an observation and record it.
        y = f(x)
        model.add_data(x, y)

        # record everything.
        info[i] = (x, y, recommender(model, bounds))

    if ftrue is not None:
        fbest = ftrue(info['xbest'])
        info = append_fields(info, 'fbest', fbest, usemask=False)

    return info
