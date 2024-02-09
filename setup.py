#!/usr/bin/env python

# How to build source distribution
#   - python setup.py sdist --format bztar
#   - python setup.py sdist --format gztar
#   - python setup.py sdist --format zip
#   - python setup.py bdist_wheel


import os

from setuptools import setup, find_packages


MAJOR = 0
MINOR = 1
MICRO = 0
VERSION = "{0}.{1}.{2}".format(MAJOR, MINOR, MICRO)


def write_version_file(fn=None):
    if fn is None:
        fn = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            os.path.join("ml_mr", "version.py"),
        )

    content = ("\n# THIS FILE WAS GENERATED AUTOMATICALLY\n"
               'ml_mr_version = "{version}"\n')

    a = open(fn, "w")
    try:
        a.write(content.format(version=VERSION))
    finally:
        a.close()


def setup_package():
    # Saving the version into a file
    write_version_file()

    setup(
        name="ml_mr",
        version=VERSION,
        description="Toolkit for Mendelian randomization using machine "
                    "learning.",
        url="https://github.com/legaultmarc/ml-mr",
        license="MIT",
        # test_suite="ml_mr.tests.test_suite",
        install_requires=["numpy >= 1.11.0", "pandas >= 0.19.0",
                          "setuptools >= 26.1.0",
                          "pytorch_lightning >= 1.0.0",
                          "scipy >= 1.9",
                          "matplotlib >= 3.7",
                          "linearmodels >= 4.0"],
        packages=find_packages(),
        classifiers=["Development Status :: 4 - Beta",
                     "Intended Audience :: Science/Research",
                     "License :: Free for non-commercial use",
                     "Operating System :: Unix",
                     "Operating System :: POSIX :: Linux",
                     "Operating System :: MacOS :: MacOS X",
                     "Operating System :: Microsoft",
                     "Programming Language :: Python",
                     "Programming Language :: Python :: 3.5",
                     "Programming Language :: Python :: 3.6",
                     "Topic :: Scientific/Engineering :: Bio-Informatics"],
        keywords="statistics causal instrumental variable genetics",
        entry_points={
            "console_scripts": [
                "ml-mr=ml_mr.cli:main"
            ]
        }
    )


if __name__ == "__main__":
    setup_package()
