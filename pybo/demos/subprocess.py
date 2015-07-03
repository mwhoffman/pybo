"""
This demo illustrates how pybo can be used to optimize a black-box
written in any other language by simply spawning off a new shell
process and running a script. In this particular example, we train
a multi-layered perceptron on the MNIST dataset, and optimize its
learning parameters using pybo.
"""
import numpy as np
import os
import os.path

import pybo
import reggie
from pybo import utils

# path to torch demos
script = os.path.join(os.environ['HOME'],
                      'torch-demos',
                      'train-a-digit-classifier',
                      'train-on-mnist.lua')

# define command
command = ' '.join([
    'th {}'.format(script),
    '--model=mlp',               # remove to use default convnet
    '--batchSize=100',
    '--epochs=1',
    '--learningRate={}',
    '--momentum={}',
    '--coefL1={}',
    '--coefL2={}',
])

# generate a black-box function from the shell command
mlp_mnist = utils.Subprocess(command)

# define bounds for each input
mlp_mnist.bounds = np.array([[0., 10.],
                             [0., 10.],
                             [0., 10.],
                             [0., 10.]], ndmin=2)

# optimize
model = reggie.make_gp(2., 10., 0.1, 0.)
info, model = pybo.solve_bayesopt(mlp_mnist, mlp_mnist.bounds, model, niter=15)
