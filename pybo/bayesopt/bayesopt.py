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
import inspect
import functools

# update a recarray at the end of solve_bayesopt.
from numpy.lib.recfunctions import append_fields

# local imports
from ..utils.random import rstate

# each method/class defined exported by these modules will be exposed as a
# string to the solve_bayesopt method so that we can swap in/out different
# components for the "meta" solver.
from . import inits
from . import solvers
from . import policies
from . import recommenders

# exported symbols
__all__ = ['solve_bayesopt']


### SOLVER COMPONENTS #########################################################

def get_components(init, policy, solver, recommender, rng):
    """
    Return model components for Bayesian optimization of the correct form given
    string identifiers.
    """
    def get_all(module, lstrip):
        """
        Get a dictionary mapping names to exported symbols from the named
        packages and strip off the named prefix to the function.
        """
        funcs = dict()
        for fname in module.__all__:
            func = getattr(module, fname)
            if fname.startswith(lstrip):
                fname = fname[len(lstrip):]
            fname = fname.lower()
            funcs[fname] = func
        return funcs

    comps = []
    parts = [(init, get_all(inits, lstrip='init_')),
             (policy, get_all(policies, lstrip='')),
             (solver, get_all(solvers, lstrip='solve_')),
             (recommender, get_all(recommenders, lstrip='best_'))]

    # construct the models in order.
    for func, mod in parts:
        func = func if hasattr(func, '__call__') else mod[func]
        kwargs_ = {}

        if 'rng' in inspect.getargspec(func).args:
            kwargs_['rng'] = rng

        if len(kwargs_) > 0:
            func = functools.partial(func, **kwargs_)

        comps.append(func)

    return tuple(comps)


### THE BAYESOPT META SOLVER ##################################################

def solve_bayesopt(f,
                   bounds,
                   T=100,
                   init='middle',
                   policy='ei',
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
    # make sure the bounds are a 2d-array.
    bounds = np.array(bounds, dtype=float, ndmin=2)

    # see if the query object itself defines ground truth.
    if (ftrue is None) and hasattr(f, 'get_f'):
        ftrue = f.get_f

    # initialize the random number generator.
    rng = rstate(rng)

    # get the model components.
    init, policy, solver, recommender = \
        get_components(init, policy, solver, recommender, rng)

    # create a list of initial points to query.
    X = init(bounds)
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
    info['xbest'][:len(Y)] = [X[np.argmax(Y[:i+1])] for i in xrange(len(Y))]

    for i in xrange(model.ndata, T):
        # get the next point to evaluate.
        index = policy(model)
        x, _ = solver(index, bounds)

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
