rmdir /s dist
python setup.py sdist
for /R %i in (dist/*.tar.gz) DO pip install %i

