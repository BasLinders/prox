from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize("conformance.pyx"),
    include_dirs=[np.get_include()] # Essential if you use NumPy arrays in Cython
)

# When launching, run this ONCE in the terminal in the repository folder: python setup.py build_ext --inplace
# The conformance file will be compiled to C.
