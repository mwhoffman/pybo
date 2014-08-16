"""
Modifications to ABC to allow for additional metaclass actions.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
from abc import ABCMeta as ABCMeta_
from abc import abstractmethod

# exported symbols
__all__ = ['ABCMeta', 'abstractmethod']


class ABCMeta(ABCMeta_):
    """
    Slight modification to ABCMeta that copies docstrings from an
    abstractmethod to its implementation if the implementation lacks a
    docstring.
    """
    def __new__(mcs, name, bases, attrs):
        abstracts = dict(
            (attr, getattr(base, attr))
            for base in bases
            for attr in getattr(base, '__abstractmethods__', set()))

        for attr, value in attrs.items():
            implements = (attr in abstracts and
                          not getattr(value, '__isabstractmethod__', False))
            if implements and not getattr(value, '__doc__', False):
                docstring = getattr(abstracts[attr], '__doc__', None)
                setattr(value, '__doc__', docstring)

        return super(ABCMeta, mcs).__new__(mcs, name, bases, attrs)
