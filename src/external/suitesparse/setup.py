from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize


# suitesparse_external = Extension(
#     name="pysuitesparse",
#     sources=["ccolamd.pyx"],
#     libraries=["ccolamd"],
#     library_dirs=["lib"],
#     include_dirs=["include"]
# )

suitesparse_external = Extension(
    name="pysuitesparse",
    sources=["ccolamd.pyx"],
    libraries=["ccolamd"],
    library_dirs=["lib"],
    include_dirs=["include"]
)

setup(
    name="pysuitesparse",
    ext_modules=cythonize([suitesparse_external])
)
