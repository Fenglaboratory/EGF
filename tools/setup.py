from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("align.pyx"),
    package_dir={"tools": ""},
)
