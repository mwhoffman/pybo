import benchfunk
import pybo


if __name__ == '__main__':

    f = benchfunk.Gramacy(0.2)
    model = pybo.init_model(f, f.bounds)
    info, model = pybo.solve_bayesopt(f, f.bounds, model, niter=10)
