"""
Base class for policy-based search strategies.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# local imports
from ..utils.abc import ABCMeta, abstractmethod

# exported symbols
__all__ = ['Policy']


class Policy:
    __metaclass__ = ABCMeta

    @abstractmethod
    def add_data(self, x, y):
        """Add a single observation of the function being optimized."""
        pass

    @abstractmethod
    def get_init(self):
        """Return an iterable, initial set of query locations"""
        pass

    @abstractmethod
    def get_next(self):
        """Get the next input point to query."""
        pass

    @abstractmethod
    def get_best(self):
        """Get the input point deemed by the policy as the best."""
        pass
