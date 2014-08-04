"""
Methods for hyperparameter inference with GPs.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
import numpy as np
import pygp.learning

# exported symbols
__all__ = ['fixed']


def fixed(gp, prior):
    return gp
