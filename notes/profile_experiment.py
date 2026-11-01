"""Profile the experiment (Part 8). Usage: uv run python notes/profile_experiment.py data/synth"""

import cProfile
import pstats
import sys
import time

from ehr_asof.experiment import TaskConfig, run
from ehr_asof.io import read_tables

tables = read_tables(sys.argv[1] if len(sys.argv) > 1 else "data/synth")
start = time.perf_counter()
cProfile.run("run(tables, TaskConfig(n_boot=1000))", "prof.out")
print(f"total {time.perf_counter() - start:.1f} s under the profiler")
pstats.Stats("prof.out").sort_stats("cumulative").print_stats(15)