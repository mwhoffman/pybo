NAME = 'pybo'
VERSION = '0.0.1'
AUTHOR = 'Matthew W. Hoffman'
AUTHOR_EMAIL = 'mwh30@cam.ac.uk'
URL = 'http://github.com/mwhoffman/pybo2'
DESCRIPTION = 'A python library for Bayesian (and bandit) optimization'


from setuptools import setup, find_packages


if __name__ == '__main__':
    setup(
        name=NAME,
        version=VERSION,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        description=DESCRIPTION,
        url=URL,
        packages=find_packages())
