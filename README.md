[![Documentation Status](https://readthedocs.org/projects/radicalpy/badge/?version=latest)](https://radicalpy.readthedocs.io/en/latest/?badge=latest)

# RadicalPy: a toolbox for radical pair spin dynamics

RadicalPy in an intuitive (object-oriented) open-source Python
framework specific to radical pair spin dynamics.

To get started take a look at the [quick start
guide](https://github.com/Spin-Chemistry-Labs/radicalpy/tree/main/docs/quick-start/guide.org)
or the
[examples](https://github.com/Spin-Chemistry-Labs/radicalpy/tree/main/examples).

The package is under constant development. [documentation](https://radicalpy.readthedocs.io/) is
up-to-date, testing partial.

If you would like to contact us, please visit the Spin Chemistry Community [Discord server](https://discord.gg/NZdjhFS3Es).

## Installation

Install simply using `pip`:

```
pip install radicalpy
```

## How to cite this software

If you used this software in your work, please cite [our paper](https://pubs.acs.org/doi/full/10.1021/acs.jctc.4c00887) (see this [bib file](CITATION.bib)).

# IMPORTANT NOTICE 2025-07-17

MANUAL INTERVENTION REQUIRED. We had some large files in the GitHub history, and to delete them, we needed to do a `git push --force`. As a consequence, if you've cloned the repo, and try `git pull`, you'll get an error about divergent branches (or something similar). What you need to do is:

```
git fetch --all       # download the repo with full history from github
git reset origin/main # or origin/<THE_BRANCH_YOU'RE_ON>
```

Git will reset your local history, to match the force-pushed history from GitHub.

**OR IF YOU'RE NOT SURE YOU CAN ALWAYS DELETE AND CLONE AGAIN!**
