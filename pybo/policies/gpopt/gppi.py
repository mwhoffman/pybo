"""
Implementation of the Probability of Improvement strategy.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
import numpy as np

# local imports
from ._base import GPAcq

# exported symbols
__all__ = ['GPPI']


class GPPI(GPAcq):
    def __init__(self, gp, bounds, xi=0.05):
        super(GPPI, self).__init__(gp, bounds)
        self.xi = xi

    def get_index(self, X):
        mu, s2 = self.gp.posterior(X)
        mu -= self.fmax + self.xi
        mu /= np.sqrt(s2, out=s2)
        return mu
