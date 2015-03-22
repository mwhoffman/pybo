import benchfunk
import reggie
import pybo


if __name__ == '__main__':
    # grab a test function and points at which to plot things
    f = benchfunk.Gramacy(0.2)
    model = reggie.BasicGP(0.2, 1.9, 0.1, -1)
    info = pybo.solve_bayesopt(f, f.bounds, model, niter=10)
