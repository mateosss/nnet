import os

from setuptools import setup
from Cython.Build import cythonize # Must be below setup import

assert os.getenv("CC"), "A c compiler must be specified through the CC envvar"

setup(
    name="DADW extension module",
    ext_modules=cythonize("dadw.pyx", annotate=True),
    zip_safe=False,
)
