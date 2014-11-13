"""
Simple utilities for creating Bayesian optimization components.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# exported symbols
__all__ = ['params']


def params(*args):
    """
    Decorator for annotating a BO component with the parameters that can be
    modified by the user.
    """
    def decorator(f):
        """Internal decorator to perform the annotation."""
        f._params = set(args)
        return f
    return decorator
