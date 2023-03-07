#! /usr/bin/env python

# run with:
#   PYTHONPATH=src python -m cProfile -o mary.profile examples/mary.py

import pstats
from pstats import SortKey

p = pstats.Stats("mary.profile")
# p.strip_dirs().sort_stats(SortKey.TIME).print_stats(30)
p.sort_stats(SortKey.TIME, SortKey.CUMULATIVE).print_stats(30)
