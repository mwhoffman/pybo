"""
Objects which implement target models for testing optimization methods.
"""

# pylint: disable=wildcard-import
from .gps import *
from .functions import *

from . import gps
from . import functions

__all__ = []
__all__ += gps.__all__
__all__ += functions.__all__
