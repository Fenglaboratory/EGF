from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("alignfix.pyx"),
    package_dir={"tools": ""},
)
