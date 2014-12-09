"""
Setup script for pybo.
"""

from setuptools import setup, find_packages

setup(name='pybo',
      version='0.0.1',
      author='Matthew W. Hoffman',
      author_email='mwh30@cam.ac.uk',
      description='A Python package for modular Bayesian optimization',
      url='http://github.com/mwhoffman/pybo',
      license='Simplified BSD',
      packages=find_packages(),
      package_data={'': ['*.txt', '*.npz']})
