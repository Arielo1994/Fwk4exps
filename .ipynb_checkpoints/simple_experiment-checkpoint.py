
import fwk4exps.speculative_monitor as fwk
f4e = fwk.SpeculativeMonitor(cpu_count=7)

def parameter_tuning(S, param, param_values):
  original_value = S.params[param]
  original_name = S.name
  params = S.params.copy()
  for value in param_values:
    params[param]=value
    if original_value == value: 
      continue
    else: 
       S2 = fwk.Strategy(original_name, S.pathExe, S.args, params)
    S = f4e.bestStrategy(S, S2)
  return S

def experimentalDesign():
    #print("experimental design2")
    S = fwk.Strategy('opt_test', 'python opt_test.py', '{x} {y}', {"x": -1.0, "y": -1.0})
    S = parameter_tuning(S, "x", [-1.0, 0.0])
    S = parameter_tuning(S, "y", [-1.0, 0.0])
    
    f4e.output = S.name + " " + str(tuple(S.params.values())) + " "

    f4e.terminate()

f4e.speculative_execution(experimentalDesign, 'instances.txt')
