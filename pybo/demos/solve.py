"""
Demo which illustrates how to use solve_bayesopt as a simple method for global
optimization. The return values are the sequence of recommendations made by the
algorithm as well as the final model. The point `xbest[-1]` is the final
recommendation, i.e. the expected maximizer.
"""

from benchfunk.functions import Sinusoidal
from pybo import solve_bayesopt


if __name__ == '__main__':
    # grab a test function and points at which to plot things
    f = Sinusoidal()
    xbest, model = solve_bayesopt(f, f.bounds, niter=30, verbose=True)
