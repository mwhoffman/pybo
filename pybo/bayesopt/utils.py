"""
Simple utilities for creating Bayesian optimization components.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
import inspect

# exported symbols
__all__ = ['params']


def params(*args):
    """
    Decorator for annotating a BO component with the parameters that can be
    modified by the user.
    """
    def decorator(f):
        """
        Internal decorator to perform the annotation.
        """
        spec = inspect.getargspec(f)
        params_valid = set(spec.args[::-1][:len(spec.defaults)])
        params = set(args)

        # make sure we're exposing valid parameters which are actually kwargs
        # in the decorated function.
        if not params.issubset(params_valid):
            raise ValueError('exposed parameters are not valid kwargs: %r'
                             % list(params - params_valid))

        # make sure we're not trying to expose rng.
        if 'rng' in params:
            raise ValueError("'rng' is a special parameter that "
                             "shouldn't be exposed")

        f._params = args
        return f

    return decorator
