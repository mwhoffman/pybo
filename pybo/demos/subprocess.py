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

from benchfunk.functions import Subprocess
import pybo
import reggie

# path to torch demos
path = os.path.join(os.environ['HOME'],
                    'torch-demos',
                    'train-a-digit-classifier')
script = os.path.join(path, 'train-on-mnist.lua')

# define command
command = ' '.join([
    'th',
    script,
    '--verbose',
    '--model=mlp',               # remove to use default convnet
    '--batchSize=100',
    '--epochs=1',
    '--learningRate={}',
    '--momentum={}',
    '--coefL1={}',
    '--coefL2={}',
])
# add torch demo repo to path
command = 'cd {}; {}; cd {};'.format(
    path,
    command,
    os.path.abspath(os.path.curdir)
)

# generate a black-box function from the shell command
mnist = Subprocess(command)

# define bounds for each input
mnist.bounds = np.array([[0., 1.],
                         [0., 1.],
                         [0., 1.],
                         [0., 1.]], ndmin=2)

# optimize
model = reggie.make_gp(1., 10., 0.1, 0., ndim=len(mnist.bounds))
info, model = pybo.solve_bayesopt(mnist, mnist.bounds, model, niter=20)

print info
