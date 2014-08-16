"""
Simple utilities for random number generation.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
import numpy as np

# exported symbols
__all__ = ['rstate']


def rstate(rng=None):
    """
    Return a numpy RandomState object. If an integer value is given then a new
    RandomState will be returned with this seed. If None is given then the
    global numpy state will be returned. If an already instantiated state is
    given this will be passed back.
    """
    if rng is None:
        rng = np.random.mtrand._rand
    elif isinstance(rng, int):
        rng = np.random.RandomState(rng)
    elif not isinstance(rng, np.random.RandomState):
        raise ValueError('unknown seed given to rstate')
    return rng
