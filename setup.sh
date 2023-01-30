#!/usr/bin/bash

rm -rf dist
python -m build
pip install dist/*tar.gz
