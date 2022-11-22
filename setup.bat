REM rmdir /s dist
REM python setup.py sdist
REM for /R %i in (dist/*.tar.gz) DO pip install %i --user
for /R %i in (dist/*.tar.gz) DO echo %i

