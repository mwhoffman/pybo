"""
Solver method for GP-based optimization which uses an inner-loop optimizer to
maximize some acquisition function, generally given as a simple function of the
posterior sufficient statistics.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import inspect
import functools

import reggie

# each method/class defined exported by these modules will be exposed as a
# string to the solve_bayesopt method so that we can swap in/out different
# components for the "meta" solver.
from . import inits
from . import solvers
from . import policies
from . import recommenders

from .utils import rstate

# exported symbols
__all__ = ['solve_bayesopt', 'init_model']


# MODEL INITIALIZATION ########################################################

def init_model(f, bounds, ninit=None, design='latin', rng=None):
    """
    Initialize model and its hyperpriors using initial data.

    Arguments:
        f: function handle
        bounds: list of doubles (xmin, xmax) for each dimension.
        ninit: int, number of design points to initialize model with.
        design: string, corresponding to a function in `pybo.inits`, with
            'init_' stripped.
        rng: int or random state.

    Returns:
        Initialized model.
    """
    rng = rstate(rng)
    bounds = np.array(bounds, dtype=float, ndmin=2)
    ndim = len(bounds)
    ninit = ninit if (ninit is not None) else 3*ndim

    # get initial design
    init_design = getattr(inits, 'init_' + design)
    xinit = init_design(bounds, ninit, rng)
    yinit = np.full(ninit, np.nan)

    # sample the initial data
    for i, x in enumerate(xinit):
        yinit[i] = f(x)

    # define initial setting of hyper parameters
    sn2 = 1e-6
    rho = yinit.max() - yinit.min() if (len(yinit) > 1) else 1.
    rho = 1. if (rho < 1e-1) else rho
    ell = 0.25 * (bounds[:, 1] - bounds[:, 0])
    bias = np.mean(yinit) if (len(yinit) > 0) else 0.

    # initialize the base model
    model = reggie.make_gp(sn2, rho, ell, bias)

    # define priors
    model.params['like.sn2'].set_prior('lognormal', -2, 1)
    model.params['kern.rho'].set_prior('lognormal', np.log(rho), 1.)
    model.params['kern.ell'].set_prior('uniform', ell / 100, ell * 10)
    model.params['mean.bias'].set_prior('normal', bias, rho)

    # initialize the MCMC inference meta-model and add data
    model.add_data(xinit, yinit)
    model = reggie.MCMC(model, n=10, burn=100, rng=rng)

    return model


# HELPER FOR CONSTRUCTING COMPONENTS ##########################################

def get_component(value, module, rng, lstrip=''):
    """
    Construct the model component if the given value is either a function
    or a string identifying a function in the given module (after stripping
    extraneous text). The value can also be passed as a 2-tuple where the
    second element includes kwargs. Partially apply any kwargs and the rng
    before returning the function.
    """
    if isinstance(value, (list, tuple)):
        try:
            value, kwargs = value
            kwargs = dict(kwargs)
        except (ValueError, TypeError):
            raise ValueError('invalid component: {:r}'.format(value))
    else:
        kwargs = {}

    if hasattr(value, '__call__'):
        func = value
    else:
        for fname in module.__all__:
            func = getattr(module, fname)
            if fname.startswith(lstrip):
                fname = fname[len(lstrip):]
            fname = fname.lower()
            if fname == value:
                break
        else:
            raise ValueError('invalid component: {:s}'.format(value))

    # get the argspec
    argspec = inspect.getargspec(func)

    # from the argspec determine the valid kwargs; these should correspond
    # to any kwargs of the function except for rng.
    if argspec.defaults is not None:
        valid = set(argspec.args[-len(argspec.defaults):])
        valid.discard('rng')
    else:
        valid = set()

    if not valid.issuperset(kwargs.keys()):
        raise ValueError("unknown arguments for {:s}: {:s}"
                         .format(func.__name__, ', '.join(kwargs.keys())))

    if 'rng' in argspec.args:
        kwargs['rng'] = rng

    if len(kwargs) > 0:
        func = functools.partial(func, **kwargs)

    return func


# FORMATTING HELPERS FOR VERBOSITY IN SOLVE_BAYESOPT ##########################

# simple format functions
int2str = '{:03d}'.format
float2str = '{: .3f}'.format


def array2str(a):
    """Formatting helper for arrays."""
    return np.array2string(a, formatter=dict(float=float2str, int=int2str))


# THE BAYESOPT META SOLVER ####################################################

def solve_bayesopt(objective,
                   bounds,
                   model=None,
                   niter=100,
                   policy='ei',
                   solver='lbfgs',
                   recommender='latent',
                   ninit=None,
                   verbose=False,
                   rng=None):
    """
    Maximize the given function using Bayesian Optimization.

    Args:
        objective: function handle representing the objective function.
        bounds: bounds of the search space as a (d,2)-array.
        model: the Bayesian model instantiation.

        niter: horizon for optimization.
        init: the initialization component.
        policy: the acquisition component.
        solver: the inner-loop solver component.
        recommender: the recommendation component.
        rng: either an RandomState object or an integer used to seed the state;
             this will be fed to each component that requests randomness.
        callback: a function to call on each iteration for visualization.

    Note that the modular way in which this function has been written allows
    one to also pass parameters directly to some of the components. This works
    for the `init`, `policy`, `solver`, and `recommender` inputs. These
    components can be passed as either a string, a function, or a 2-tuple where
    the first item is a string/function and the second is a dictionary of
    additional arguments to pass to the component.

    Returns:
        A numpy record array containing a trace of the optimization process.
        The fields of this array are `x`, `y`, and `xbest` corresponding to the
        query locations, outputs, and recommendations at each iteration. If
        ground-truth is known an additional field `fbest` will be included.
    """
    rng = rstate(rng)
    bounds = np.array(bounds, dtype=float, ndmin=2)

    # get modular components.
    policy = get_component(policy, policies, rng)
    solver = get_component(solver, solvers, rng, lstrip='solve_')
    recommender = get_component(recommender, recommenders, rng, lstrip='best_')

    # initialize model
    if model is None:
        model = init_model(objective, bounds, ninit, design='latin', rng=rng)
    else:
        # copy the model in order to avoid overwriting
        model = model.copy()

    # allocate a datastructure containing algorithm progress
    xbest = list()

    # Bayesian optimization loop
    for i in xrange(niter):
        if model.ndata == 0:
            # if the model has no data that means that we were given a model,
            # but that model had no initial data selected. So just fall back on
            # a very simple initialization scheme.
            x = inits.init_middle(bounds)[0]
        else:
            # get the next point to evaluate.
            index = policy(model, bounds)
            x, _ = solver(index, bounds)

        # make an observation and record it.
        y = objective(x)
        model.add_data(x, y)
        xbest += [recommender(model, bounds)]

        # print out the progress if requested.
        if verbose:
            print('i={:s}, x={:s}, y={:s}, xbest={:s}'
                  .format(int2str(i),
                          array2str(x),
                          float2str(y),
                          array2str(xbest[-1])))

    xbest = np.array(xbest, ndmin=2)

    return xbest, model
