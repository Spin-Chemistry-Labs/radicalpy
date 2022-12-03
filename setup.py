#!/usr/bin/env python

from pathlib import Path

from setuptools import find_packages, setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="radicalpy",
    version="0.5.1",
    license="MIT",
    author="Lewis M. Antill",
    author_email="lewismantill@gmail.com",
    maintainer="Emil Vatai",
    maintainer_email="emil.vatai@gmail.com",
    packages=find_packages("src"),
    package_dir={"": "src"},
    url="https://github.com/Spin-Chemistry-Labs/radicalpy",
    keywords="simulation spin-dynamics radical-pair",
    install_requires=["numpy", "scipy", "matplotlib", "scikit-learn"],
    include_package_data=True,
    long_description=long_description,
    long_description_content_type="text/markdown",
)

# Build with:
# python setup.py sdist
#
# Local install with:
# pip install dist/*.tar.gz --user
#
# Upload:
# twine upload dist/*
#
# Create tag/release to upload from github
