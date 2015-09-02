import benchfunk
import pybo


if __name__ == '__main__':
    f = benchfunk.Sinusoidal()
    info, model = pybo.solve_bayesopt(f, f.bounds, niter=30, verbose=True)
