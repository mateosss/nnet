import os

from setuptools import setup
from Cython.Build import cythonize # Must be below setup import

assert os.getenv("CC"), "A c compiler must be specified through the CC envvar"

setup(
    name="cynet_native",
    ext_modules=cythonize("nets/cynet_native.pyx", annotate=True),
    zip_safe=False,
)
