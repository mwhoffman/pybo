"""
Various utility functions.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import re
import subprocess

__all__ = ['rstate', 'SubprocessQuery', 'InteractiveQuery']


def rstate(rng=None):
    """
    Return a RandomState object. This is just a simple wrapper such that if rng
    is already an instance of RandomState it will be passed through, otherwise
    it will create a RandomState object using rng as its seed.
    """
    if not isinstance(rng, np.random.RandomState):
        rng = np.random.RandomState(rng)
    return rng


class SubprocessQuery(object):
    """
    Class for black-boxes that should be run from the shell. Simply pass the
    shell command with variables replaced with `{}` with python string
    formatting specs inside, then call the object with inputs to replace the `{}`
    in the same order as in the provided string.
    """
    def __init__(self, command):
        self.command = command

    def __call__(self, x):
        out = subprocess.check_output(self.command.format(*x), shell=True)
        out = out.splitlines()[-1]                      # keep last line
        out = re.compile(r'\x1b[^m]*m').sub('', out)    # strip color codes
        out = out.split('=')[-1]                        # strip left hand side
        return np.float(out)


class InteractiveQuery(object):
    """
    Wrapper for queries which interactively query the user.
    """
    def __init__(self, prompt='Enter value at design x = {}\ny = '):
        self.prompt = prompt

    def __call__(self, x):
        y = input(self.prompt.format(x))
        if not isinstance(y, (np.int, np.long, np.float)):
            # FIXME: this should probably just re-query the user rather than
            # raising an exception.
            raise ValueError('output must be a number')
        return y
