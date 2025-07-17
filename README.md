[![Documentation Status](https://readthedocs.org/projects/radicalpy/badge/?version=latest)](https://radicalpy.readthedocs.io/en/latest/?badge=latest)

# IMPORTANT NOTICE

MANUAL INTERVENTION REQUIRED. We had some large files in the GitHub history, and to delete them, we needed to do a `git push --force`. As a consequence, if you've cloned the repo, and try `git pull`, you'll get an error about divergent branches (or something similar). What you need to do is:

```
git fetch --all       # download the repo with full history from github
git reset origin/main # or origin/<THE_BRANCH_YOU'RE_ON>
```

Git will reset your local history, to match the force-pushed history from GitHub.

**OR IF YOU'RE NOT SURE YOU CAN ALWAYS DELETE AND CLONE AGAIN!**

# RadicalPy: a toolbox for radical pair spin dynamics

RadicalPy in an intuitive (object-oriented) open-source Python
framework specific to radical pair spin dynamics.

To get started take a look at the [quick start
guide](https://github.com/Spin-Chemistry-Labs/radicalpy/tree/main/docs/quick-start/guide.org)
or the
[examples](https://github.com/Spin-Chemistry-Labs/radicalpy/tree/main/examples).

The package is still under development. Basic functionality is
implemented, [documentation](https://radicalpy.readthedocs.io/) is
sparse, testing partial.

If you would like to contact us, please visit the Spin Chemistry Community [Discord server](https://discord.io/spin-chemistry-community/).

## Installation

Install simply using `pip`:

```
pip install radicalpy
```
