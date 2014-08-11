NAME = 'pybo'
VERSION = '0.0.1'
AUTHOR = 'Matthew W. Hoffman'
AUTHOR_EMAIL = 'mwh30@cam.ac.uk'
URL = 'http://github.com/mwhoffman/pybo2'
DESCRIPTION = 'A python library for Bayesian (and bandit) optimization'


from setuptools import setup, find_packages, Extension
from numpy.distutils.misc_util import get_numpy_include_dirs
import os, subprocess


class SimpleExtension(Extension):
    def __init__(self, *sources):
        psources = []
        for source in sources:
            name, ext = os.path.splitext(source)
            if ext == '.pyx':
                subprocess.call(['cython', source])
                psources.append(name + '.c')
            else:
                psources.append(source)

        name, ext = os.path.splitext(psources[0])
        name = name.replace(os.path.sep, '.')

        Extension.__init__(self, name,
                           sources=psources,
                           include_dirs=get_numpy_include_dirs())


if __name__ == '__main__':
    setup(
        name=NAME,
        version=VERSION,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        description=DESCRIPTION,
        url=URL,
        zip_safe=False,
        packages=find_packages(),
        ext_modules=[
            SimpleExtension('pybo/globalopt/direct.pyx'),
        ])
