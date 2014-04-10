"""
Base class for GP acquisition functions.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
import numpy as np
import abc

# local imports
from ..__base import Policy

# exported symbols
__all__ = ['GPAcq']


class GPAcq(Policy):
    def __init__(self, gp, bounds):
        self.gp = gp
        self.bounds = np.array(bounds, ndmin=2, copy=False)
        self.xmax = None
        self.fmax = -np.inf

    def add_data(self, x, y):
        # add data to the model.
        self.gp.add_data(x, y)

        # find the incumbent.
        mu, _ = self.gp.posterior(self.gp._X)
        i = mu.argmax()
        self.xmax = self.gp._X[i]
        self.fmax = mu[i]

        # FIXME: I don't like the above code since it requires knowledge of the
        # underlying GP structure. it may also be easier and cheaper to do this
        # in the object itself since we've already computed the covariance
        # between gp._X and itself.

    @abc.abstractmethod
    def get_index(self, X):
        pass

    def get_next(self):
        # FIXME! Right now this is "implemented" but will return an error, just
        # so I can instantiate children classes. This NEEDS TO BE FIXED! :) In
        # particular it should call DIRECT or something once I integrate that
        # into the code.
        raise NotImplementedError

    def get_best(self):
        return self.xmax
