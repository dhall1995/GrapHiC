import io
import os

from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
from numpy import get_include as get_numpy_include
from os import path

exts = [Extension("dtrack_utils", [path.join("GrapHiC/utils", "dtrack_utils.pyx")],
                  include_dirs=[get_numpy_include()])
]


VERSION = None
with io.open(
    os.path.join(os.path.dirname(__file__), "GrapHiC/__init__.py"),
    encoding="utf-8",
) as f:
    for l in f:
        if not l.startswith("__version__"):
            continue
        VERSION = l.split("=")[1].strip(" \"'\n")
        break
PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))

REQUIREMENTS_FILE = os.path.join(PROJECT_ROOT, "requirements.txt")

with open(REQUIREMENTS_FILE) as f:
    install_reqs = f.read().splitlines()

install_reqs.append("setuptools")

setup(
    name="GrapHiC",
    version="0.0.1",
    description="A tool to generate bespoke Graph datasets from HiC and ChIPseq",
    author="Dominic Hall & Arian Jamasb",
    author_email="dh486@cam.ac.uk",
    url="https://github.com/dhall1995/GrapHiC-ML",
    packages=find_packages()+['dtrack_utils'],
    package_data={
        "": ["LICENSE.txt", "README.md", "requirements.txt", "*.csv"]
    },
    include_package_data=True,
    install_requires=install_reqs,
    ext_modules = cythonize(exts),
    license="MIT",
    platforms="any",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Development Status :: 5 - Production/Stable",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering",
    ],
    long_description="""
This package provides functionality for producing bespoke graph datasets using multi-omics data and Hi-C. We provide compatibility with the standard bigwig and bed UCSC data formats and integrate with the popular Cooler format for Hi-C data.
Contact
=============
If you have any questions or comments about GrapHiC,
please feel free to contact me via
email: dh486@cam.ac.uk
This project is hosted at https://github.com/dhall1995/GrapHiC-ML
""",
)
