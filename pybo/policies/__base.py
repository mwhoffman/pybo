"""
Base class for continuous BO search strategies.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
import abc

# exported symbols
__all__ = ['Policy']


class Policy:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def add_data(self, x, y):
        pass

    @abc.abstractmethod
    def get_next(self):
        pass

    @abc.abstractmethod
    def get_best(self):
        pass
