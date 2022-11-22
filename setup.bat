REM rmdir /s dist
REM python setup.py sdist
for /R %i in (dist/*.tar.gz) DO echo %i
REM pip install %i --user

