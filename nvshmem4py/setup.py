import atexit
import glob
import os
import shutil
import sys
import tempfile

from Cython.Build import cythonize
from setuptools import setup, Extension, find_packages
from packaging.version import Version
import Cython

# Set package name dynamically
PACKAGE_NAME = os.environ.get("PACKAGE_NAME")

ext_modules = [
    "nvshmem.bindings.nvshmem"
]

def calculate_modules(module):
    module = module.split(".")

    lowpp_mod = module.copy()
    lowpp_mod_pyx = os.path.join(*module[:-1], f"{module[-1]}.pyx")
    lowpp_mod = ".".join(lowpp_mod)
    lowpp_ext = Extension(
        lowpp_mod,
        sources=[lowpp_mod_pyx],
        language="c++",
    )

    cy_mod = module.copy()
    cy_mod[-1] = f"cy{cy_mod[-1]}"
    cy_mod_pyx = os.path.join(*cy_mod[:-1], f"{cy_mod[-1]}.pyx")
    cy_mod = ".".join(cy_mod)
    cy_ext = Extension(
        cy_mod,
        sources=[cy_mod_pyx],
        language="c++",
    )

    inter_mod = module.copy()
    inter_mod.insert(-1, "_internal")
    inter_mod_pyx = os.path.join(*inter_mod[:-1], f"{inter_mod[-1]}.pyx")
    inter_mod = ".".join(inter_mod)
    inter_ext = Extension(
        inter_mod,
        sources=[inter_mod_pyx],
        language="c++",
    )

    return lowpp_ext, cy_ext, inter_ext


# Note: the extension attributes are overwritten in build_extension()
ext_modules = [e for ext in ext_modules for e in calculate_modules(ext)]


compiler_directives = {"embedsignature": True, "show_performance_hints": False}

setup(
    name=PACKAGE_NAME,
    ext_modules=cythonize(ext_modules, verbose=True, language_level=3, compiler_directives=compiler_directives),
    zip_safe=False,
    packages=find_packages(include=["nvshmem", "nvshmem.*"]),
    include_package_data=True,
    options={"build_ext": {"inplace": True}},
    install_requires=open(f"{os.path.dirname(__file__)}/requirements.txt").read().splitlines()
)
