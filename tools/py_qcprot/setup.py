from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        "py_qcprot",
        sources=["py_qcprot.pyx", "qcprot.c"],
        include_dirs=[np.get_include()],
    )
]
setup(
    name="py_qcprot",
    ext_modules=cythonize(ext_modules),
    include_dirs=[np.get_include()],
)
