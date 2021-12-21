from strategy import Strategy
from speculative_monitor import SpeculativeMonitor

import itertools
import sys
import numpy as np

strategies_file = sys.argv[1]
counters_file = sys.argv[2]

f4e = SpeculativeMonitor(15, None, 'instances.txt',strategies_file=strategies_file, counters_file=counters_file)

total=0
for str_name in Strategy.strategy_dict:
  algo=Strategy.strategy_dict[str_name]
  total+=algo.n_runs
  if algo.est_means is not None and len(algo.est_means)>0:
    print(algo.params, np.mean(algo.results), np.std(algo.results), algo.n_runs )
print (total)

for str_name in Strategy.strategy_dict:
  algo=Strategy.strategy_dict[str_name]
  print(algo.params, np.mean(algo.results[0:10]))

for c in f4e.counters:
  for str_name in Strategy.strategy_dict:
    algo=Strategy.strategy_dict[str_name]
    if algo.params == c[1]:
      print(algo.params, np.mean(algo.results[0:c[4]]))
      break

for c in f4e.counters:
  print(c[0])

for c in f4e.counters:
  print(c[1],c[4])

for c in f4e.counters:
  if len(c[2])>8:
    print(c[3],c[2][8])
