from strategy import Strategy
from speculative_monitor import SpeculativeMonitor


## The experimental design
def parameter_tuning(S, param, param_values):
    original_value = S.params[param]
    original_name = S.name
    params = S.params.copy()
    for value in param_values:
        params[param]=value
        if original_value == value: 
            continue
        else: 
            S2 = Strategy.create_strategy(original_name, S.exe, S.params_str, params)
        S = f4e.bestStrategy(S, S2)
    return S

def experimentalDesign():
    S = Strategy.create_strategy('BSG_CLP', './BSG_CLP', '--alpha={a} --beta={b} --gamma={g} -p {p} -t 30', {"a": 0.0, "b": 0.0, "g": 0.0, "p": 0.0})
    f4e.output = ""
    S = parameter_tuning(S, "a", [0.0, 1.0, 2.0, 4.0, 8.0])
    f4e.output = str(S.params["a"]) + " "
    S = parameter_tuning(S, "b", [0.0, 0.5, 1.0, 2.0, 4.0])
    f4e.output += str(S.params["b"]) + " "
    S = parameter_tuning(S, "g", [0.0, 0.1, 0.2, 0.3, 0.4])
    f4e.output += str(S.params["g"]) + " " 
    S = parameter_tuning(S, "p", [0.00, 0.01, 0.02, 0.03, 0.04])
    f4e.output += str(S.params["p"]) + " "
    f4e.terminate(f4e)  


f4e = SpeculativeMonitor(15, experimentalDesign, 'instancesCLP-shuf.txt', strategies_file="strategies.dat", counters_file="counters.dat")

total=0
for str_name in Strategy.strategy_dict:
  algo=Strategy.strategy_dict[str_name]
  total+=algo.n_runs
  if algo.est_means is not None and len(algo.est_means)>0:
    print(algo.params, algo.n_runs)
print (total)
len(f4e.counters)

f4e.speculative_execution(strategies_file=None, counters_file=None)

