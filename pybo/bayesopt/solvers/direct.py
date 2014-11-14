"""
Interface to the nlopt DIRECT implementation.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# by default export nothing.
__all__ = []

try:
    # try and import nlopt, and if not this package will not define or export
    # anything.
    import nlopt
    import numpy as np

    # exported symbols
    __all__ += ['solve_direct']

    def solve_direct(f, bounds):
        def objective(x, grad):
            """Objective function in the form required by nlopt."""
            if grad.size > 0:
                fx, gx = f(x[None], grad=True)
                grad[:] = gx[0][:]
            else:
                fx = f(x[None], grad=False)
            return fx[0]

        bounds = np.array(bounds, ndmin=2)

        opt = nlopt.opt(nlopt.GN_DIRECT_L, bounds.shape[0])
        opt.set_lower_bounds(list(bounds[:, 0]))
        opt.set_upper_bounds(list(bounds[:, 1]))
        opt.set_ftol_rel(1e-6)
        opt.set_max_objective(objective)

        xmin = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) / 2
        xmin = opt.optimize(xmin)
        fmax = opt.last_optimum_value()

        return xmin, fmax

except ImportError:
    pass
