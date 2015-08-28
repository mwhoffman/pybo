"""
Various utility functions.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np

__all__ = ['repr_args', 'rstate']


def rstate(rng=None):
    """
    Return a RandomState object. This is just a simple wrapper such that if rng
    is already an instance of RandomState it will be passed through, otherwise
    it will create a RandomState object using rng as its seed.
    """
    if not isinstance(rng, np.random.RandomState):
        rng = np.random.RandomState(rng)
    return rng
