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
import warnings
import mwhutils.random as random

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
    def get_func(key, value, module, lstrip):
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
                raise ValueError('invalid arguments for component %r' % key)
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
                raise ValueError('invalid identifier for component %r' % key)

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
            raise ValueError("unknown keyword arguments for component '{:s}'"
                             .format(key))

        if 'rng' in argspec.args:
            kwargs['rng'] = rng

        if len(kwargs) > 0:
            func = functools.partial(func, **kwargs)

        return func

    return (get_func('init', init, inits, lstrip='init_'),
            get_func('policy', policy, policies, lstrip=''),
            get_func('solver', solver, solvers, lstrip='solve_'),
            get_func('recommender', recommender, recommenders, lstrip='best_'))


### THE BAYESOPT META SOLVER ##################################################

def solve_bayesopt(objective,
                   bounds,
                   model,
                   niter=100,
                   init='latin',
                   policy='ei',
                   solver='lbfgs',
                   recommender='latent',
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
    # make a copy of model
    model = model.copy()

    # make sure the bounds are a 2d-array.
    bounds = np.array(bounds, dtype=float, ndmin=2)

    # initialize the random number generator.
    rng = random.rstate(rng)

    # get the model components.
    init, policy, solver, recommender = \
        get_components(init, policy, solver, recommender, rng)

    # allocate a datastructure containing "convergence" info.
    info = np.zeros(niter,
                    [('x', np.float, (len(bounds),)),
                     ('y', np.float),
                     ('xbest', np.float, (len(bounds),))])
    ninit = 0

    # initialize model
    if model.ndata == 0:
        # create a list of initial points to query.
        X = init(bounds)

        if len(X) > niter:
            # warn if initialization goes over budget
            msg = 'initialization samples exceeded evaluation budget `niter`'
            warnings.warn(msg, stacklevel=2)

            # truncate initial samples to budget
            X = X[:niter]

        Y = [objective(x) for x in X]


        for i, (x, y) in enumerate(zip(X, Y)):
            model.add_data(x, y)

            # record everything.
            info[i] = (x, y, recommender(model, bounds))

        # deduct initial points from requested number of iterations
        ninit = model.ndata

    # Bayesian optimization loop
    for i in xrange(ninit, niter):
        # get the next point to evaluate.
        index = policy(model, bounds)
        x, _ = solver(index, bounds)

        # make an observation and record it.
        y = objective(x)
        model.add_data(x, y)

        # record everything.
        info[i] = (x, y, recommender(model, bounds))

    return info, model
