"""
Sample from low-discrepancy sequences.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
import numpy as np

# exported symbols
__all__ = ['random']


def random(bounds, n):
    """
    Sample n points uniformly at random from the specified region, given by
    a list of [(lo,hi), ..] bounds in each dimension.
    """
    bounds = np.array(bounds, ndmin=2, copy=False)
    d = len(bounds)
    X = bounds[:,0] + np.random.rand(n,d) * (bounds[:,1] - bounds[:,0])
    return X
