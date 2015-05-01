import benchfunk
import reggie
import pybo


if __name__ == '__main__':
    # grab a test function and points at which to plot things
    f = benchfunk.Gramacy(0.2)
    model = reggie.make_gp(0.2, 1.9, 0.1, -1)
    info, model = pybo.solve_bayesopt(f, f.bounds, model, niter=10)
