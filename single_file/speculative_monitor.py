import multiprocessing
import numpy as np
import copy
import random
import pickle
import pymc3 as pm
import logging
from strategy import Strategy

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

#discard_extreme: (simulation) number of samples that are discarded when the optimistic or pessimistic mean is computed for each strategy
#cut_impact: minimo impacto que debe tener una estrategia para ser seleccionada como volatile strategy sin importar impacto de estrategias siguientes
class SpeculativeMonitor:
  def __init__(self, cpu_count, experimental_design, pifile, strategies_file=None, counters_file=None, with_base_strategy=True, discard_extreme=4, cut_impact=0.5):
      self.counters=[]
      Strategy.strategy_dict=dict()

      self.cpu_count = cpu_count
      
      """recibe la funcion de diseño experimental y
      la ruta del archivo de instancias, setea estos datos"""
      self.experimental_design = experimental_design
      self.pifile = pifile
      self.base_strategy = None
      self.with_base_strategy = with_base_strategy
      self.discard_extreme=discard_extreme
      self.cut_impact=cut_impact

      # read data
      print("reading file of instances")
      with open(pifile) as f:
          self.instances = f.readlines()

      if strategies_file is not None and counters_file is not None:
        Strategy.strategy_dict=load(strategies_file)
        self.counters=load(counters_file)
        if with_base_strategy:
          for algo_str in Strategy.strategy_dict:
              self.base_strategy = Strategy.strategy_dict[algo_str]
              break

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


  def _select_strategy(self, n=100, iter=0):
    self._simulations(n=n)
    mid_counter = copy.deepcopy(self.state_counter)
    print("counters:",[mid_counter[s][0] if s in mid_counter else 0 for s,_,_ in self.tree_descent_path ])
   
    ## se comparan estrategias usando likelihood de nodo más lejano de la raiz con P>10% (probable_state)
    probable_index = len(self.tree_descent_path)-1

    for s, _, _ in reversed(self.tree_descent_path):
        ini_value=mid_counter[s][0] if s in mid_counter else 0
        if s in mid_counter and mid_counter[s][0]>=10: break
        probable_index -= 1

    volatile_strategy = None
    max_impact= -100.0
    evaluated = set()

    for state, S1, S2 in self.tree_descent_path:

        if state not in mid_counter: continue 
        
        for alg_base in [S1,S2]:
            if alg_base is None: continue
            if alg_base in evaluated: continue
            if alg_base.n_runs == len(self.instances): continue
            

            self.simul_mean[alg_base] = np.partition(alg_base.est_means,-self.discard_extreme-1)[-self.discard_extreme-1] #np.max(alg_base.est_means)  10/250
            self._simulations(n, alg_base=alg_base)
            opt_counter = copy.deepcopy(self.state_counter)
            opt_a=[opt_counter[s][0] if s in opt_counter else 0 for s,_,_ in self.tree_descent_path]


            self.simul_mean[alg_base] = np.partition(alg_base.est_means,self.discard_extreme)[self.discard_extreme] #np.min(alg_base.est_means)
            self._simulations(n, alg_base=alg_base)
            pes_a=[self.state_counter[s][0]  if s in self.state_counter else 0  for s,_,_ in self.tree_descent_path]

            ## se calcula peor likelihood de probable_state 
            val = np.minimum(opt_a[probable_index],pes_a[probable_index])

            ## estrategia escogida será la que tiene un gran impacto en likelihood y se encuentra en niveles tempranos del árbol
            impact = 1.0 - (val/ini_value)

            if impact>max_impact:
                max_impact = impact
                volatile_strategy=alg_base

            print(alg_base.params,impact)
            
            evaluated.add(alg_base)

            if max_impact > self.cut_impact: break

        if max_impact > self.cut_impact: break
        

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
            if self.strategies[0].results is None: self._execute(self.strategies[0])
            if self.strategies[1].results is None: self._execute(self.strategies[1])
            executions = True
            if self.with_base_strategy and self.base_strategy == None:
              self.base_strategy = self.strategies[0]
              print("base strategy:", self.base_strategy.params)
          

      done = True
      for state, S1, S2 in self.tree_descent_path:
        if S1 is None: break
        if S1.n_runs < len(self.instances) or S2.n_runs < len(self.instances):
            done = False
            break
            
      print(self.output)
      print("######end_tree_descent########")
      return done
      #print("node at the end of tree_descent:", node)


  def estimate_means(self):
      for k in Strategy.strategy_dict:
        alg = Strategy.strategy_dict[k]

        if alg.results is not None and alg.needs_to_be_sampled:
          print("Sampling",alg.params)
          n = alg.n_runs
          res=alg.results[0:n]
          if self.base_strategy is not None:
            res -= self.base_strategy.results[0:n]

          alg.est_means = SpeculativeMonitor.sample_means(res, len(self.instances)-len(res))
          alg.needs_to_be_sampled=False

  def speculative_execution(self, strategies_file=None, counters_file=None):
        i=0
        while True:
            done = self._tree_descent()
            if done: break

            probable_output = self.output
            self.estimate_means()
            strategy, diff, counter = self._select_strategy(n=100, iter=i)
            if strategy==None: break
            
            print([(S1.n_runs, S2.n_runs) for state, S1, S2 in self.tree_descent_path[:-1]])
            print([np.partition(S1.est_means,-5)[-5]-np.partition(S1.est_means,4)[4] for state, S1, S2 in self.tree_descent_path[:-1]])

            print("selected strategy:",strategy.params, diff)
            self._execute(strategy)

            if  self.base_strategy is not None and self.base_strategy.run_instances() <  strategy.run_instances():
                self._execute(self.base_strategy)

            total_runs=0
            for str_name in Strategy.strategy_dict:
                algo=Strategy.strategy_dict[str_name]
                total_runs+=algo.n_runs

            self.counters.append((probable_output, strategy.params, counter, total_runs, strategy.n_runs))
            print("total runs:", total_runs)

            if strategies_file is not None:
                save(strategies_file,Strategy.strategy_dict)
            
            if counters_file is not None:
                save(counters_file,self.counters)
            i+=1

  @staticmethod
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
            sigma = pm.Uniform('sigma', lower=0.001, upper=20)

            returns = pm.Normal('returns', mu=mu, sd=sigma, observed=data)

            #step = pm.Metropolis()
            #trace = pm.sample(250, step, cores=4, progressbar=False, tune=500)

            step = pm.NUTS() # Hamiltonian MCMC with No U-Turn Sampler
            trace = pm.sample(250, step,  tune=1000, cores=8, random_seed=123, progressbar=False, )

            for t in trace: 
                __sum = np.random.normal(c * t["mu"], np.sqrt(c) * t["sigma"])
                mean = (sum(data) + __sum)/(len(data)+c)
                means.append(mean)
            
        return means