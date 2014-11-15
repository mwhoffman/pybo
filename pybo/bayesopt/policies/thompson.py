"""
Implementation of Thompson sampling for continuous spaces.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

from collections import deque
from ..utils import params

__all__ = ['Thompson']


@params('n')
def Thompson(model, n=100, rng=None):
    """
    Implementation of Thompson sampling for continuous models using a finite
    approximation to the kernel matrix with `n` Fourier components.
    """
    if hasattr(model, '__iter__'):
        model = deque(model, maxlen=1).pop()
    return model.sample_fourier(n, rng).get
