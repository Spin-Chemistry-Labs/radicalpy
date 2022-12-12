# Some notes for us (the developers)

- Follow [Google style docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
- Use `black`

# Building

## On Linux

Build with:
```python
python setup.py sdist
```

Local install with:
```bash
pip install dist/*.tar.gz --user
```

Upload:
```bash
twine upload dist/*
```

Create tag to upload to PyPI.
