#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="radicalpy",
    version="0.1b",
    license="MIT",
    author="Lewis M. Antill",
    author_email="lewismantill@gmail.com",
    maintainer="Emil Vatai",
    maintainer_email="emil.vatai@gmail.com",
    packages=find_packages("src"),
    package_dir={"": "src"},
    url="https://github.com/Spin-Chemistry-Labs/radicalpy",
    keywords="quantum spin chemistry",
    install_requires=[
        "numpy",
    ],
)
