"""
Implementation of Thompson sampling for continuous spaces.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
import numpy as np

# local imports
from ._base import GPAcq

# not "exactly" local...
from pygp.extra.fourier import FourierSample

# exported symbols
__all__ = ['Thompson']


class Thompson(GPAcq):
    def __init__(self, gp, bounds):
        super(Thompson, self).__init__(gp, bounds)
        self.n = 250
        self.f = FourierSample(self.gp, self.n)

    def add_data(self, x, y):
        super(Thompson, self).add_data(x, y)
        self.f = FourierSample(self.gp, self.n)

    def get_index(self, X):
        return self.f(X)
