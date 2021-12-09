from strategy import Strategy
from speculative_monitor import SpeculativeMonitor

import itertools
import sys
sd = sys.argv[1]

def experimentalDesign():
    base_params = {"x": -1.0, "y": -1.0}
    S = Strategy.create_strategy('opt_test', 'python3.6 opt_test.py '+sd, '{x} {y}', base_params)

    #factorial design
    f4e.output = ""
    for x,y in itertools.product(*[[-1.0, 0.0, 1.0],[-1.0, 0.0, 1.0]]):
        params =  {"x": x, "y": y}
        if params == base_params: continue
        S2 = Strategy.create_strategy('opt_test', 'python3.6 opt_test.py '+sd,  '{x} {y}', params)
        S = f4e.bestStrategy(S, S2)

    f4e.output += str(S.params)
    f4e.terminate()  


if __name__ ==  '__main__':
    f4e = SpeculativeMonitor(4, experimentalDesign, 'instances.txt')

    total=0
    for str_name in Strategy.strategy_dict:
        algo=Strategy.strategy_dict[str_name]
        total+=algo.n_runs
        if algo.est_means is not None and len(algo.est_means)>0:
            print(algo.params, algo.n_runs)
    print (total)
    len(f4e.counters)

    f4e.speculative_execution(strategies_file="fd-strategies-"+sd+".dat", counters_file="fd-counters-"+sd+".dat")

