"""
Implementation of the GPUCB strategy.
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
__all__ = ['GPUCB']


class GPUCB(GPAcq):
    def __init__(self, gp, bounds, delta=0.1, xi=0.2):
        super(GPUCB, self).__init__(gp, bounds)

        # dimensionality of the state-space.
        d = len(self.bounds)

        # note that as per Srinivas et al. (among other experiments) I'm scaling
        # the beta term by a fifth.
        self.a = xi*2*np.log(np.pi**2 / 3 / delta)
        self.b = xi*(4+d)

    def get_index(self, X):
        mu, s2 = self.gp.posterior(X)
        beta = self.a + self.b * np.log(self.gp.ndata+1)
        return mu + np.sqrt(beta*s2)
