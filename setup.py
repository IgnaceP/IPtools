from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("snap_polygon.pyx", language_level = "3"),
)
