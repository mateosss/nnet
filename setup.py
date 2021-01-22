from setuptools import setup
from Cython.Build import cythonize
from distutils import sysconfig

sysconfig.get_config_vars()['CC'] = 'gcc-8'

setup(
    name='DADW extension module',
    ext_modules=cythonize("dadw.pyx", annotate=True),
    zip_safe=False,
)
