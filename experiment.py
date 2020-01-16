
import fwk4exps.speculative_monitor as fwk
import numpy as np 

f4e = fwk.SpeculativeMonitor(cpu_count=10)

def factorial_design(S, *values, param_names=[]):
  configurations = np.array(np.meshgrid(*values)).T.reshape(-1,len(values))
  original_params_str = str(tuple(S.params.values()))
  params = S.params.copy()
  for conf in configurations:
    i=0
    for p in param_names:
      params[p] = conf[i]; i+=1
      if original_params_str == str(tuple(params.values())):
         continue
      else:
         S2 = fwk.Strategy(S.name, S.pathExe, S.args, params)
      S = f4e.bestStrategy(S, S2)
  return S
      

def parameter_tuning(S, param, param_values):
  params = S.params.copy()
  for value in param_values:
    params[param]=value
    S2 = fwk.Strategy(S.name, S.pathExe, S.args, params)
    S = f4e.bestStrategy(S, S2)
  return S

def experimentalDesign():
    S = fwk.Strategy('opt_test', 'python opt_test.py', '{x} {y}', {"x": -1.0, "y": -1.0})
    S = parameter_tuning(S, "x", [-1.0, 0.0, 1.0])
    S = parameter_tuning(S, "y", [-1.0, 0.0, 1.0])
    
    f4e.output = S.name + " " + str(tuple(S.params.values())) 

    f4e.terminate()

def experimentalDesign2():
    S = fwk.Strategy('opt_test', 'python opt_test.py', '{x} {y}', {"x": -1.0, "y": -1.0})
    S = factorial_design(S, [-1.0, -0.5, 0.0, 0.5, 1.0], [-1.0, -0.5, 0.0, 0.5, 1.0], param_names=["x","y"])
    
    f4e.output = S.name + " " + str(tuple(S.params.values())) 

    f4e.terminate()
    
def bsg_clp():
    S = fwk.Strategy('opt_test', './BSG_CLP', '--alpha={a} --beta={b} --gamma={c} -p {p} -t 30', {'a':0.0, 'b':0.0, 'c':0.0, 'p':0.0})
    S = parameter_tuning(S, 'a', [1.0, 2.0, 4.0, 8.0])
    S = parameter_tuning(S, 'b', [1.0, 2.0, 4.0, 8.0])
    S = parameter_tuning(S, 'c', [0.1, 0.2, 0.3, 0.4])
    S = parameter_tuning(S, 'p', [0.01, 0.02, 0.03, 0.04])
    
    f4e.output = S.name + " " + str(tuple(S.params.values())) 

    f4e.terminate()

f4e.speculative_execution(bsg_clp, 'instancesCLP-shuf.txt')