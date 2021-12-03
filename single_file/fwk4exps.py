import pymc3 as pm
import random
import logging

def sample_means(data, c):
    # https://twiecki.github.io/blog/2015/11/10/mcmc-sampling/
    # print(data)
    # print("sampling parameters given data")
    np.random.seed(123)

    logger = logging.getLogger('pymc3')
    logger.setLevel(logging.ERROR)
    _logger = logging.getLogger("theano.gof.compilelock")
    _logger.setLevel(logging.ERROR)

    means = [] 
    sample = []
    # with suppress_stdout:
    with pm.Model():
        mu = pm.Normal('mu', np.mean(data), 1)
        sigma = pm.Uniform('sigma', lower=0.001, upper=2)

        returns = pm.Normal('returns', mu=mu, sd=sigma, observed=data)

        #step = pm.Metropolis()
        #trace = pm.sample(250, step, cores=4, progressbar=False, tune=500)

        step = pm.NUTS() # Hamiltonian MCMC with No U-Turn Sampler
        trace = pm.sample(250, step,  tune=500, cores=4, random_seed=123, progressbar=False, )

        for t in trace: 
            __sum = np.random.normal(c * t["mu"], np.sqrt(c) * t["sigma"])
            mean = (sum(data) + __sum)/(len(data)+c)
            means.append(mean)
          
    return means


## STRATEGY

import subprocess
import multiprocessing

class Strategy:
    strategy_dict = dict()

    def get_id(self):
      return str(tuple(params.values())) + " " + exe + " " + name

    @staticmethod
    def create_strategy(name, exe, params_str, params):
      str_id = str(tuple(params.values())) + " " + exe + " " + name
      if str_id in Strategy.strategy_dict:
        return Strategy.strategy_dict[str_id]
      else:
        S = Strategy(name, exe, params_str, params)
        Strategy.strategy_dict[str_id]=S
        return S

    def __init__(self, name, exe, params_str, params):
      self.name=name
      self.exe=exe
      self.params_str=params_str
      self.params=params
      self.results=None
      self.est_means=None
      self.needs_to_be_sampled=False
      self.n_runs=0
      self.no_impact=1


    def run(self, instance, i, PI, ret_dic):
      aux = copy.copy(PI)
      aux = aux.split('/')
      aux.pop()
      if len(aux)>0 :
        aux.pop(0)
        PI = ""
        for e in aux:
            PI = PI+"/"+e
        PI = PI+"/"
        instance = PI+instance

      commando = self.exe + " " + instance.rstrip() + " " + self.params_str.format(**self.params)
      #print(commando)
      output = subprocess.getoutput(commando)
      output = output.splitlines()
      ret_dic[i] = float(output[-1])  

    def run_instances(self):
      return self.n_runs

    def norm_results(self, base_strategy):
      n = self.run_instances()
      res = self.results[0:n] - base_strategy.results[0:n]
      return res


### Speculative Monitor and Staff

import multiprocessing
import numpy as np
import copy

class SpeculativeMonitor:
  def __init__(self, cpu_count, experimental_design, pifile):
      self.cpu_count = cpu_count
      
      """recibe la funcion de diseño experimental y
      la ruta del archivo de instancias, setea estos datos"""
      self.experimental_design = experimental_design
      self.pifile = pifile
      self.base_strategy = None

      # read data
      print("reading file of instances")
      with open(pifile) as f:
          self.instances = f.readlines()

import traceback, sys
class TraceBackInfo(object):
    def getExperimentState():
        """
        Funcion para obtener el 'Estado' de una funcion
        Se debe tener en cuenta el numero de llamadas a funcion
        previos a esta funcion en este caso -3 indica que hace 3 funciones
        se llamo a  la funcion a la cual le queremos sacar su estado.
        el estado se compone de sus variables locales y el numero de linea
        donde fue llamado, estos valores son transformados
        en un string que se retorna para formar la llave del
        nodo.
        """
        #print("getexperiment info")
        #  ----------Traceback info:
        extracted_list = traceback.extract_stack()
        formated_traceback_list = traceback.format_list(extracted_list)
        #  ----------Formated traceback list
        important_line = formated_traceback_list[-3]
        #print("important_line:")
        #print(important_line)
        #print("line_no:")
        line_no = extracted_list[-3][1]
        #print(line_no)

        #print("local variables from experimentalDesign:")
        call_frame = sys._getframe(2)
        eval_locals = call_frame.f_locals
        #print(eval_locals)

        return_str = str(line_no)+str(eval_locals)

        return return_str


def _execute(self, alg):
    """ dado un algoritmo ejecuta cierto numero de
    instancias y guarda los resultados en la matris de
    resultados globales"""

    if alg.results is None: alg.results = np.zeros((len(self.instances)))

    manager = multiprocessing.Manager()
    jobs = []
    return_dict =  manager.dict()
    
    #print ( "executing:"+ str(self.cpu_count))
    runs=0
    for i in range(0, self.cpu_count):
        instance_index = alg.n_runs+i
        if instance_index >= len(self.instances):
            break

        runs+=1

        instance = self.instances[instance_index]
        p = multiprocessing.Process(target=alg.run, args=(instance, instance_index, self.pifile, return_dict))
        jobs.append(p)

    for p in jobs: p.start()
    for p in jobs: p.join()

    keys = [key for key, value in return_dict.items()]
    
    for k in keys:
        alg.results[k]= return_dict[k]

    alg.needs_to_be_sampled = True
    alg.n_runs += runs

def bestStrategy(self, S1, S2):
    state = TraceBackInfo.getExperimentState()

    if S1.results is None or S2.results is None:
      self.strategies = (S1,S2)
      raise ValueError

    ret = S2
    if self.simulation==False:
      self.tree_descent_path.append((state,S1,S2))
      if np.mean(S1.norm_results(self.base_strategy)) > np.mean(S2.norm_results(self.base_strategy)): ret= S1
    else:
      if state in self.state_counter: self.state_counter[state][0] += 1
      else: self.state_counter[state] = [1,self.depth, self.output]
      self.depth +=1

      if self.simul_mean[S1] > self.simul_mean[S2]: ret= S1

    return ret

def terminate(self):
    state = TraceBackInfo.getExperimentState()
    if self.simulation==True:
      if state in self.state_counter: self.state_counter[state][0] += 1
      else: self.state_counter[state] = [1, self.depth, self.output]
    else:
      self.tree_descent_path.append((state,None,None))

def _simulations(self, n=1000, alg_base=None):
    self.simulation=True
    self.state_counter = dict()
    
    if not hasattr(self, 'simul_mean'): self.simul_mean = dict()
    
    for count in range(n):
      for alg_str in Strategy.strategy_dict:
          alg=Strategy.strategy_dict[alg_str]
          if alg.est_means is None: continue

          if alg_base is None or alg != alg_base: self.simul_mean[alg] = alg.est_means[random.randrange(0, len(alg.est_means))]
          
      try:
          self.strategies = None #almacena estrategias en caso de comparacion sin data
          self.depth = 0
          self.experimental_design()
      except ValueError as x:
          pass

import random


def _select_strategy(self, n=100, iter=0):
    _simulations(self, n=n)
    mid_counter = copy.deepcopy(self.state_counter)
    print("counters:",[mid_counter[s][0] if s in mid_counter else 0 for s,_,_ in self.tree_descent_path ])

    volatile_strategy = None
    max_impact= 0.0
    evaluated = set()
    
    ## se comparan estrategias usando likelihood de nodo más lejano de la raiz con P>10% (probable_state)
    probable_index = len(self.tree_descent_path)-1

    for s, _, _ in reversed(self.tree_descent_path):
      ini_value=mid_counter[s][0] if s in mid_counter else 0
      if s in mid_counter and mid_counter[s][0]>=10: break
      probable_index -= 1
      
  

    while volatile_strategy==None:

      no_impact = []
      for state, S1, S2 in self.tree_descent_path:
        if state not in mid_counter: continue 
        
        impact = None
        for alg_base in [S1,S2]:
          if alg_base is None: continue
          if alg_base in evaluated: continue
          if iter % alg_base.no_impact != 0: continue
          
          self.simul_mean[alg_base] = np.partition(alg_base.est_means,-10)[-10] #np.max(alg_base.est_means)  10/250
          _simulations(self, n, alg_base=alg_base)
          opt_counter = copy.deepcopy(self.state_counter)
          opt_a=[opt_counter[s][0] if s in opt_counter else 0 for s,_,_ in self.tree_descent_path]


          self.simul_mean[alg_base] = np.partition(alg_base.est_means,9)[9] #np.min(alg_base.est_means)
          _simulations(self, n, alg_base=alg_base)
          pes_a=[self.state_counter[s][0]  if s in self.state_counter else 0  for s,_,_ in self.tree_descent_path]

          ## se calcula peor likelihood de probable_state 
          val = np.minimum(opt_a[probable_index],pes_a[probable_index])

          ## estrategia escogida será la que tiene un gran impacto en likelihood y se encuentra en niveles tempranos del árbol
          impact = 1.0 - (val/ini_value)

          if impact < 0.8:
            no_impact.append(alg_base)
          else:
            alg_base.no_impact = 1

          if impact>max_impact:
            max_impact = impact
            volatile_strategy=alg_base

          print(alg_base.params,val/ini_value)
          

          evaluated.add(alg_base)

        if max_impact > 0.8: break

      if iter >0 and max_impact > 0.8:
        for algo in no_impact:
          algo.no_impact *= 2

      iter=0
      max_impact=-1.0
      

    #si volatile is None, retornar primera estrategia con n_runs < len(instances)
    volatile_strategy.no_impact=1

    return volatile_strategy, max_impact, [mid_counter[s][0] if s in mid_counter else 0 for s,_,_ in self.tree_descent_path ]



def _tree_descent(self):
    """ Desciende por el arbol seleccionando la rama de la
    estrategia con mas probabilidades de ganar.
    si no hay suficientes instances ejecutadas, se corren las
    instances necesarias y se continua descendiendo hasta
    llegar a un nodo hoja """
    print("######start_tree_descent2########")

    self.simulation=False
    reach_leave = False
    executions = False
    while reach_leave == False:
      try:
        self.tree_descent_path=[]
        self.strategies = None #almacena estrategias en caso de comparacion sin data
        self.experimental_design()
        reach_leave = True
      except ValueError as x:
        if executions == True and len(self.tree_descent_path)>2: break
        if self.strategies is not None:
          if self.strategies[0].results is None: _execute(self, self.strategies[0])
          if self.strategies[1].results is None: _execute(self, self.strategies[1])
          executions = True
          if self.base_strategy == None:
            self.base_strategy = self.strategies[0]
            print("base strategy:", self.base_strategy.params)
        

          

    print(self.output)
    print("######end_tree_descent########")
    #print("node at the end of tree_descent:", node)


def estimate_means(self):
    for k in Strategy.strategy_dict:
      alg = Strategy.strategy_dict[k]

      if alg.results is not None and alg.needs_to_be_sampled:
         print("Sampling",alg.params)
         n = alg.n_runs
         res = alg.results[0:n] - self.base_strategy.results[0:n]

         alg.est_means = sample_means(res, len(self.instances)-len(res))
         alg.needs_to_be_sampled=False


import pickle 
def save(filename, data):
  #save state
  outfile = open(filename,'wb')
  pickle.dump(data ,outfile)

  outfile.close()

def load (filename):
  infile = open(filename,'rb')
  data = pickle.load(infile)
  infile.close()
  return data


def load_experiments():
  f4e = SpeculativeMonitor(15, experimentalDesign, 'instancesCLP-shuf.txt')
  Strategy.strategy_dict=load("strategies.dat")
  counters=load("counters.dat")

  for algo_str in Strategy.strategy_dict:
    f4e.base_strategy = Strategy.strategy_dict[algo_str]
    break
  return f4e, counters


import time



def speculative_execution(f4e, counters):
  while True:
    _tree_descent(f4e)
    probable_output = f4e.output
    estimate_means(f4e)
    strategy, diff, counter = _select_strategy(f4e, n=100, iter=i)
    
    print([(S1.n_runs, S2.n_runs) for state, S1, S2 in f4e.tree_descent_path[:-1]])
    print([np.partition(S1.est_means,-10)[-10]-np.partition(S1.est_means,9)[9] for state, S1, S2 in f4e.tree_descent_path[:-1]])

    print("selected strategy:",strategy.params, diff)
    _execute(f4e, strategy)

    if f4e.base_strategy.run_instances() <  strategy.run_instances():
      _execute(f4e, f4e.base_strategy)

    total_runs=0
    for str_name in Strategy.strategy_dict:
      algo=Strategy.strategy_dict[str_name]
      total_runs+=algo.n_runs

    counters.append((probable_output, strategy.params, counter, total_runs))
    print("total runs:", total_runs)

    save("strategies.dat",Strategy.strategy_dict)
    save("counters.dat",counters)
    if counter[len(counter)-1] >= 99: break
    #time.sleep(2)
    clear_output()


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
        S = bestStrategy(f4e, S, S2)
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
    terminate(f4e)  

#counters=[]
#Strategy.strategy_dict=dict()
#f4e = SpeculativeMonitor(25, experimentalDesign, 'instancesCLP-shuf.txt')

f4e, counters = load_experiments()

total=0
for str_name in Strategy.strategy_dict:
  algo=Strategy.strategy_dict[str_name]
  total+=algo.n_runs
  if algo.est_means is not None and len(algo.est_means)>0:
    print(algo.params, algo.n_runs)
print (total)
len(counters)

speculative_execution(f4e, counters)

