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

# local imports
from ..utils import params

# exported symbols
__all__ = ['Thompson']


@params('n')
def Thompson(model, _, n=100):
    """
    Implementation of Thompson sampling for continuous models using a finite
    approximation to the kernel matrix with `n` Fourier components.
    """
    if hasattr(model, '__iter__'):
        model = deque(model, maxlen=1).pop()
    return model.sample_fourier(n).get
