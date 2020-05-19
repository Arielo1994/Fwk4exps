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

    S = fwk.Strategy('BSG_CLP', './BSG_CLP', '--alpha={a} --beta={b} --gamma={g} -p {p}', {"a": 0.0, "b": 0.0, "g": 0.0, "p": 0.0})

    S = parameter_tuning(S, "a", [0.0, 1.0, 2.0, 4.0, 8.0])
    S = parameter_tuning(S, "b", [0.0, 0.5, 1.0, 2.0, 4.0])
    S = parameter_tuning(S, "g", [0.0, 0.1, 0.2, 0.3, 0.4])
    S = parameter_tuning(S, "p", [0.0, 0.1, 0.2, 0.3, 0.4])

    f4e.output = S.name + " " + str(tuple(S.params.values())) + " "
    f4e.terminate()    

f4e.speculative_execution(experimentalDesign, 'instancesCLP-shuf.txt')
