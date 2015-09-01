import benchfunk
import pybo


if __name__ == '__main__':
    f = benchfunk.Gramacy(0.2)
    info, model = pybo.solve_bayesopt(f, f.bounds, niter=10)
