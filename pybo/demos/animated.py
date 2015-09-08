"""
Animated demo showing progress of Bayesian optimization on a simple
(but highly multimodal) one-dimensional function.
"""

import matplotlib.pyplot as plt
import numpy as np

from reggie import make_gp, MCMC
from benchfunk import Gramacy

from pybo import inits
from pybo import policies
from pybo import solvers
from pybo import recommenders


# we shouldn't be doing this. but I'll leave it for now.
plt.rc('lines', linewidth=2.0)
plt.rc('legend', scatterpoints=3)
plt.rc('figure', facecolor='white')
plt.rc('axes', grid=True)
plt.rc('grid', color='k', linestyle='-', alpha=0.2, linewidth=0.5)
plt.rc('axes', color_cycle=[
    (0.21568627450980393, 0.49411764705882355, 0.7215686274509804),
    (0.30196078431372547, 0.6862745098039216, 0.2901960784313726),
    (0.596078431372549, 0.3058823529411765, 0.6392156862745098),
    (1.0, 0.4980392156862745, 0.0),
    (0.6509803921568628, 0.33725490196078434, 0.1568627450980392),
    (0.9686274509803922, 0.5058823529411764, 0.7490196078431373)])


from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.projections import register_projection


class CustomAxes(Axes):
    name = 'custom_defaults'

    def cla(self):
        super(CustomAxes, self).cla()
        self.spines['top'].set_visible(False)
        self.spines['right'].set_visible(False)
        self.xaxis.set_tick_params(direction='out', top=False)
        self.yaxis.set_tick_params(direction='out', right=False)

    def scatter(self, x, y, **kwargs):
        kwargs_ = dict(s=40, c='k', linewidths=1.5, marker='o', 
                       facecolors='none')
        kwargs_.update(kwargs)
        super(CustomAxes, self).scatter(x, y, **kwargs_)

    def plot_banded(self, x, y=None, b1=None, **kwargs):
        if y is None:
            lines = self.plot(x, **kwargs)
        else:
            lines = self.plot(x, y, **kwargs)
        x = lines[0].get_xdata()
        y = lines[0].get_ydata()
        c = lines[0].get_color()
        a = lines[0].get_alpha()
        a = 1.0 if (a is None) else a
        if b1 is None:
            self.fill_between(x, 0, y, color=c, alpha=0.2*a)
        else:
            self.fill_between(x, y-b1, y+b1, color=c, alpha=0.2*a)

    def axvline(self, *args, **kwargs):
        kwargs_ = dict(color='r', linestyle='--')
        kwargs_.update(kwargs)
        super(CustomAxes, self).axvline(*args, **kwargs_)

    def axhline(self, *args, **kwargs):
        kwargs_ = dict(color='r', linestyle='--')
        kwargs_.update(kwargs)
        super(CustomAxes, self).axhline(*args, **kwargs_)


register_projection(CustomAxes)


if __name__ == '__main__':
    # grab a test function and points at which to plot things
    f = Gramacy()
    bounds = f.bounds

    # get initial data and some test points.
    X = inits.init_latin(bounds, 3)
    Y = np.array([f(x_) for x_ in X])

    # initialize the model
    model = make_gp(0.01, 1.9, 0.1, 0)
    model.add_data(X, Y)

    # set a prior on the parameters
    model.params['like.sn2'].set_prior('uniform', 0.005, 0.015)
    model.params['kern.rho'].set_prior('lognormal', 0, 100)
    model.params['kern.ell'].set_prior('lognormal', 0, 10)
    model.params['mean.bias'].set_prior('normal', 0, 20)

    # make a model which samples parameters
    model = MCMC(model, n=20, rng=None)
    fbest = list()
    x = np.linspace(bounds[0][0], bounds[0][1], 500)

    # create a new figure
    fig = plt.figure(figsize=(10, 6))
    fig.set_tight_layout(dict(w_pad=12))

    while True:
        # get index to solve it and plot it
        index = policies.EI(model, bounds, 0.1)

        # get the recommendation and the next query
        xbest = recommenders.best_observed(model, bounds)
        xnext, _ = solvers.solve_lbfgs(index, bounds)

        # evaluate the posterior and the acquisition function
        mu, s2 = model.predict(x[:, None])
        alpha = index(x[:, None])
        fbest += [f.get_f(xbest)]
        xdata = model.data[0].ravel()
        ydata = model.data[1]

        fig.clear()
        ax1 = plt.subplot2grid((2, 2), (0, 0), projection='custom_defaults')
        ax2 = plt.subplot2grid((2, 2), (1, 0), projection='custom_defaults', sharex=ax1)
        ax3 = plt.subplot2grid((2, 2), (0, 1), projection='custom_defaults', rowspan=2)

        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax2.get_yticklabels(), visible=False)
        plt.setp(ax2.get_yaxis().get_offset_text(), visible=False)

        # plot the posterior and data
        ax1.set_title('current model')
        ax1.plot_banded(x, mu, 2*np.sqrt(s2), label='posterior')
        ax1.scatter(xdata, ydata, label='data')
        ax1.axvline(xbest, label='recomm.')
        ax1.legend(loc=3, bbox_to_anchor=(1, 0))

        # plot the acquisition function
        ax2.set_title('current policy')
        ax2.plot_banded(x, alpha, label='acquisition')
        ax2.axvline(xnext, label='next query')
        ax2.set_xlim(*bounds)
        ax2.legend(loc=3, bbox_to_anchor=(1, 0))

        # plot the latent function at recomended points
        ax3.set_title('value of recommendation')
        ax3.axhline(f.get_f(f.xopt))
        ax3.plot(fbest)

        # draw
        fig.canvas.draw()

        # add the next evaluation
        model.add_data(xnext, f(xnext))
