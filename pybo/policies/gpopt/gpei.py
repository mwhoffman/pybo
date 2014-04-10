"""
Implementation of the Expected Improvement strategy.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
import numpy as np
import scipy.stats as ss

# local imports
from ._base import GPAcq

# exported symbols
__all__ = ['GPEI']


class GPEI(GPAcq):
    def __init__(self, gp, bounds, xi=0.0):
        super(GPEI, self).__init__(gp, bounds)
        self.xi = xi

    def get_index(self, X):
        mu, s2 = self.gp.posterior(X)
        s = np.sqrt(s2, out=s2)
        d = mu - self.fmax - self.xi
        z = d / s
        return d*ss.norm.cdf(z) + s*ss.norm.pdf(z)
