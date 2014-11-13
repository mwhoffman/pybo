"""
Acquisition functions based on (GP) UCB.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# use this to simplify (slightly) the Thompson implementation with sampled
# models.
from collections import deque

# exported symbols
__all__ = ['Thompson']


def Thompson(model, N=100):
    if hasattr(model, '__iter__'):
        model = deque(model, maxlen=1).pop()
    return model.sample_fourier(N).get
