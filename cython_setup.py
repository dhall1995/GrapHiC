# This script can be run using the following command:
#
#    python cython_setup.py build_ext --inplace
#

from setuptools import setup, Extension
from Cython.Build import cythonize
from numpy import get_include as get_numpy_include
from os import path

exts = [Extension("dtrack_utils", [path.join("GrapHiC/utils", "dtrack_utils.pyx")],
                  include_dirs=[get_numpy_include()])
]

setup(
    name="GrapHiC",
    version="0.0.1",
    description="A tool to generate bespoke Graph datasets from HiC and ChIPseq",
    packages=['dtrack_utils','link_utils'],
    ext_modules = cythonize(exts)
)

