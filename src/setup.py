from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize("conformance.pyx"),
    include_dirs=[np.get_include()] # Essential if you use NumPy arrays in Cython
)
