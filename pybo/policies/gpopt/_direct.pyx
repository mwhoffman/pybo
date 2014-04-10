"""
Code for performing global optimization using the DIRECT method. See the
`solve_direct` method. Also see "Lipschitzian Optimization Without the Lipschitz
Constant", (Jones et al, 1993) for more details.
"""

# make sure the python code uses float division.
from __future__ import division

# global imports.
import numpy as np

# make cython and numpy available to the cython code.
cimport cython
cimport numpy as np

# standard math routines.
from libc.math cimport sqrt
from libc.math cimport fabs as abs
from libc.math cimport fmax as max
from libc.math cimport fmin as min

__all__ = ['solve_direct']


#===============================================================================
# main entry point for the DIRECT method. this basically just sets up everything
# and passes responsibility off to _run_direct.
#

def solve_direct(f, bounds, nmax=10000, maxit=50, epsilon=1e-10, report=False):
    """
    solve_direct(f, bounds, [nmax, maxit, epsilon, report])

    Perform DIRECT on the given function f over a space where dimension i is
    bounded by bounds[i][0] and bounds[i][1]

    Optional parameters:
        nmax: maximum number of samples.
        maxit: number of iterations.
        epsilon: desired accuracy.
        report: whether or not to report intermediate results.
    """
    # initialize the rectangle structures.
    f, trans, x, ell, r, fx = _init_rects(f, bounds, nmax)

    # run the inner loop, which just continually divides up the rectangles,
    # inserting them into the rects structure, and returns when it reaches nmax.
    xmin, fmin, n = _run_direct(f, x, ell, r, fx, epsilon, maxit)

    # by default return a tuple of xmin, f(xmin). make sure to translate
    # xmin back into the original space first.
    rval = (trans(xmin), fmin)

    # if we want extra info add a dictionary containing the bounds of
    # each rectangle and the function evaluation in the center of that
    # rect.
    if report:
        extra = dict()
        extra['lb'] = trans(x[:n] - ell[:n] / 2.0)
        extra['ub'] = trans(x[:n] + ell[:n] / 2.0)
        extra['x'] =  trans(x[:n])
        extra['fx'] = fx[:n]
        rval += (extra,)

    return rval


# just initialize everything. this only really needs to be called once, so we
# don't need it to be fast at all. so python is fine. (plus it has no loops!)

def _init_rects(f, bounds, nmax):
    """
    Initialize a set of rectangles inside a bounding box with upper/lower bounds
    given by the `bounds`. Preallocate room for nmax rectangles (with 2*dim
    wiggle room).

    Returns (x, ell, r, fx, ftrans, trans).
    """
    bounds = np.array(bounds, ndmin=2, copy=False)
    lb = bounds[:,0]
    ub = bounds[:,1]
    ndim = len(lb)
    nmax += 2*ndim

    # allocate the structures.
    x = np.empty((nmax, ndim))
    ell = np.empty((nmax, ndim))
    r = np.empty(nmax)
    fx = np.empty(nmax)

    # create lambdas for evaluating f(x) in the original space, and also
    # for translating x back into this space.
    trans = lambda x: (ub - lb)*x + lb
    ftrans = lambda x: f(trans(x))

    # initialize the first point. note that we're translating each
    # dimension into the range [0,1] and ftrans is taking care of
    # translating it back into the original space.
    ell[0] = np.ones_like(ub)
    x[0] = 0.5 * ell[0]
    r[0] = 0.5 * np.sqrt(ndim)
    fx[0] = ftrans(x[0,None])

    return ftrans, trans, x, ell, r, fx


#===============================================================================
# main *fast* loop for performing DIRECT. this should be pretty fast and pretty
# much all C other than the call into python to evaluate the function. this will
# loop through and figure out which rectangle to evaluate next and then call
# _divide_rect in order to do that.
#

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef _run_direct(object f,
                  np.ndarray[np.float_t, ndim=2] x,
                  np.ndarray[np.float_t, ndim=2] ell,
                  np.ndarray[np.float_t, ndim=1] r,
                  np.ndarray[np.float_t, ndim=1] fx,
                  float epsilon, int maxit):
    cdef int n = 1
    cdef int imin = 0
    cdef int nmax = x.shape[0] - 2*x.shape[1]
    cdef int i, j
    cdef float Kmin, Kmax

    cdef np.ndarray[np.int_t, ndim=1] props = np.empty(nmax, dtype=np.int)
    cdef int it, nprop

    for it in xrange(maxit):
        nprop = 0
        for j in xrange(n):
            # FIXME: use real values here.
            Kmin = 999999999
            Kmax = -Kmin
            for i in xrange(n):
                # don't compare a rectangle to itself.
                if i == j: continue
                if r[i] != r[j]:
                    K = (fx[j] - fx[i]) / (r[j] - r[i])
                    if r[i] < r[j]:
                        Kmax = max(Kmax, K)
                    else:
                        Kmin = min(Kmin, K)
                        if Kmin <= 0.0: break
                elif fx[j] > fx[i]:
                    # the jth rectangle cannot be "potentially optimal"
                    # if there exists any other rectangle with the same
                    # d_i and smaller f_i.
                    break

                if Kmax > -np.inf and Kmin < np.inf and Kmin < Kmax:
                    break

            # NOTE: this else is attached to the for loop, so we'll check
            # this condition (and as a result possibly append j to the list
            # of proposed rectangles) only if we loop through everything
            # above without breaking.
            else:
                if (Kmin == np.inf or
                    fx[j] - Kmin*r[j] <= fx[imin] - epsilon*np.abs(fx[imin])):
                    props[nprop] = j
                    nprop += 1

        for j in xrange(nprop):
            i = props[j]
            nnew = _divide_rect(f, i, n, x, ell, r, fx)
            inew = n + fx[n:n+nnew].argmin()
            imin = inew if (fx[inew] < fx[imin]) else imin
            n += nnew
            if n >= nmax:
                return x[imin], fx[imin], n

    return x[imin], fx[imin], n


#===============================================================================
# when we've found a rectangle we like, divide it up! this acts both to divide
# the rectangle up and to evaluate the rectangle inside the new pieces.
#

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int _divide_rect(object f, int p, int n,
                       np.ndarray[np.float_t, ndim=2] x,
                       np.ndarray[np.float_t, ndim=2] ell,
                       np.ndarray[np.float_t, ndim=1] r,
                       np.ndarray[np.float_t, ndim=1] fx):
    cdef int i, j, k
    cdef int d = x.shape[1]
    cdef int m = 0
    cdef float ellmax = 0
    cdef float ellnew

    cdef np.ndarray[np.int_t, ndim=1] dims = np.empty(d, dtype=np.int)

    # find the maximum dimension.
    for i in range(d):
        ellmax = max(ellmax, ell[p,i])

    # just save the new size of these dimensions.
    ellnew = ellmax / 3.0

    # find every dimension coinciding with the max.
    for i in range(d):
        if ell[p,i] == ellmax:
            dims[m] = i
            m += 1

    # create the centers for each of our new evaluation points.
    for k in range(m):
        i = n + 2*k
        k = dims[k]
        for j in range(d):
            x[i,j] = x[i+1,j] = x[p,j]
        x[i,k] -= ellnew
        x[i+1,k] += ellnew

    # NOTE: this should be our only call into python, so we're going to
    # take a hit in three places, first in the two slicing calls to x
    # and fx and finally in the vectorized call to f.
    fx[n:n+2*m] = f(x[n:n+2*m])

    cdef np.ndarray[np.float_t, ndim=1] fmax = np.empty(m, dtype=np.float)
    cdef np.ndarray[np.int_t, ndim=1] indices

    # get the maximum along each split dimension.
    for k in range(m):
        i = n + 2*k
        fmax[k] = max(fx[i], fx[i+1])

    # find the indices sorting each dimension in increasing order of
    # their fmax value.
    indices = fmax.argsort()

    # divide each rectangle, computing the length of the rectangle as
    # well as the radius as we go along.
    for k in reversed(indices):
        i = n + 2*k
        ell[p, dims[k]] = ellnew
        r[i] = 0.0
        for j in range(d):
            ell[i,j] = ell[i+1,j] = ell[p,j]
            r[i] += ell[i,j] ** 2
        r[i] = r[i+1] = 0.5 * sqrt(r[i])

    # compute the new radius of the split rectangle.
    r[p] = 0.0
    for j in range(d):
        r[p] += ell[p,j] ** 2
    r[p] = 0.5 * sqrt(r[p])

    return 2*m
