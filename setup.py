NAME = 'pybo'
VERSION = '0.1.dev3'
AUTHOR = 'Matthew W. Hoffman'
AUTHOR_EMAIL = 'mwh30@cam.ac.uk'


def setup_package(parent_package='', top_path=None):
    from setuptools import find_packages
    from numpy.distutils.core import setup, Extension
    import os, subprocess

    class SimpleExtension(Extension):
        def __init__(self, source):
            name, ext = os.path.splitext(source)
            if ext == '.pyx':
                subprocess.call(['cython', source])
                sources = [name + '.c']
            else:
                sources = [source]
            name = name.replace(os.path.sep, '.')
            Extension.__init__(self, name, sources=sources)

    setup(
        name=NAME,
        version=VERSION,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        packages=find_packages(),
        zip_safe=False,
        ext_modules=[
            SimpleExtension('pybo/policies/_direct.pyx'),
        ])


if __name__ == '__main__':
    setup_package()
