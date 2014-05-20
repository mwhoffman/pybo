=====
pybo2
=====

*This is the beginnings of a poorly put together version of various snippets of
other BO code I've written. As such it will (hopefully!) slowly replicate and
extend that code, but in the meanwhile this code exists so that I can actually
get things done. (Note the 2 above!)*

`pybo` is a python package for bandit (and Bayesian) optimization.

This package provides methods for performing optimization under bandit feedback.
Really, all this means is that we have some function *f(x)* that we want to
optimize, but rather than knowing this function analytically we can only query
it, or perhaps some other *f'(x)* which corresponds to a noise-corrupted version
of *f*.

