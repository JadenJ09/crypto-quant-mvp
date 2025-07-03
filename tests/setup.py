from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extension = Extension(
    "indicators_cython",
    ["indicators_cython.pyx"],
    include_dirs=[numpy.get_include()]
)

setup(
    ext_modules=cythonize([extension])
)
