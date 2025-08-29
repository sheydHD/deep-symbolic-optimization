from setuptools import setup, Extension
import numpy
from Cython.Build import cythonize

setup(
    ext_modules=cythonize([
        Extension(
            "dso.cyfunc",
            ["dso_pkg/dso/cyfunc.pyx"],
            include_dirs=[numpy.get_include()],
            extra_compile_args=["-O3"],
        )
    ]),
    zip_safe=False,
)
