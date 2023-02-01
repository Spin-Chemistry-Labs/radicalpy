echo ON
rmdir /s dist
python -m build
for /R %%i in (dist\*.tar.gz) DO pip install %%i

