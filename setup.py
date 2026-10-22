"""Portable Cython build; performance-affecting options are explicit."""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup


def enabled(name: str) -> bool:
    value = os.environ.get(name, "0")
    if value not in {"0", "1"}:
        raise RuntimeError(f"{name} must be 0 or 1, received {value!r}")
    return value == "1"


openmp = enabled("PAGERANK_OPENMP")
native = enabled("PAGERANK_NATIVE")
compile_args: list[str] = []
link_args: list[str] = []
include_dirs = [np.get_include()]
library_dirs: list[str] = []

if sys.platform == "win32":
    compile_args.append("/O2")
    if openmp:
        compile_args.append("/openmp")
    if native:
        warnings.warn("PAGERANK_NATIVE is ignored by the Windows/MSVC build")
else:
    compile_args.append("-O3")
    if native:
        compile_args.append("-march=native")
    if openmp and sys.platform == "darwin":
        compile_args += ["-Xpreprocessor", "-fopenmp"]
        link_args.append("-lomp")
        prefix = os.environ.get("PAGERANK_OPENMP_PREFIX")
        if prefix:
            include_dirs.append(os.path.join(prefix, "include"))
            library_dirs.append(os.path.join(prefix, "lib"))
            link_args.append(f"-Wl,-rpath,{os.path.join(prefix, 'lib')}")
    elif openmp:
        compile_args.append("-fopenmp")
        link_args.append("-fopenmp")

extension = Extension(
    "pagerank_lab._kernels",
    ["src/pagerank_lab/_kernels.pyx"],
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    define_macros=[
        ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
        ("PAGERANK_HAVE_OPENMP", "1" if openmp else "0"),
        ("PAGERANK_NATIVE_BUILD", "1" if native and sys.platform != "win32" else "0"),
    ],
    extra_compile_args=compile_args,
    extra_link_args=link_args,
)

setup(
    ext_modules=cythonize(
        [extension],
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
            "cdivision": True,
        },
        annotate=enabled("PAGERANK_ANNOTATE"),
    )
)

