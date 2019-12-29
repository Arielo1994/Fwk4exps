#  class imports
from .classes.tree import Tree
from .classes.sampler import Sampler
from .classes.plotter import Plotter
from .classes.traceback_info import TraceBackInfo
from .classes.strategy import Strategy
from .classes.node import Node
from .classes.leaf_node import LeafNode
from .classes.anytime import Anytime
# utilities imports
import multiprocessing
import datetime
import os
import hashlib
import numpy as np


class SpeculativeMonitor(object):
    """
    Clase principal, ejecuta el algoritmo de ejecucion especulativa
    y la entrega de datos anytime.
    policy: max_sim o descent_spec
    """
    def __init__(self, cpu_count=None, policy="max_sim"):
        self.anytime = False
        self.__count = None
        self.__msg = None
        self.__speculativeNode = None
        self.tree = Tree()
        self.pifile = None
        self.experimental_design = None
        self.instances = None
        self.global_results = None
        self.__totalSimulations = 5000
        self.iteration = 1
        # self.__numOfExecutions = 0
        self.s2id = {}
        self.s_id = 0
        self.the_end = False
        # self.algoritmos = dict()
        self.node_dict = dict()
        self.leaf_dict = dict()
        self.quality_animation = None
        self.parameter_histogram = None
        self.quality_frame = None
        self.sampler = Sampler()
        self.instances = None
        self.cpu_count = cpu_count
        self.optimisticQuality = dict()
        self.pessimisticQuality = dict()
        self.amplitude = dict()
        self.experiment_hash = None
        self.tree_descent_outcome = None
        self.policy = policy
        self.tree_desc_likelihood = 0
        self.max_sim_likelihood = 0
        self._tree_descent_strategies = {}
        self.execution_num = 0
        self.opt_res = {}
        self.pes_res = {}
        if cpu_count is None:
            self.cpu_count = multiprocessing.cpu_count()

    def initialize(self, experimental_design, pifile):
        """recibe la funcion de diseño experimental y
        la ruta del archivo de instancias, setea estos datos"""
        self.experimental_design = experimental_design
        self.pifile = pifile
        # read data
        print("reading file of instances")
        with open(pifile) as f:
            self.instances = f.readlines()
        self.load_permutation_file()

    def load_permutation_file(self):
        """
        Carga archivo de permutacion.
        """
        print("load_permutation_file")
        # obtiene hash de archivo de instancias
        # el hash sera el nombre de la carpeta del experimento
        # en la carpeta se guardara archivo de permutacion
        # (orden de instancias)
        hasher = hashlib.md5()
        with open(self.pifile, 'rb') as afile:
            buf = afile.read()
            hasher.update(buf)
        file_md5_hash = hasher.hexdigest()
        self.experiment_hash = file_md5_hash
        Strategy.permutation_folder = file_md5_hash

        print("Strategy.permutation_folder:", Strategy.permutation_folder)

        # checka si existe una carpeta para este archivo de instancias
        if not os.path.exists('results/'+file_md5_hash):
            print("creating permutation folder!")
            # si no existe creo la carpeta y el archivo de permutacion
            os.makedirs('results/'+file_md5_hash+'/strategies')
            with open(self.pifile) as f:
                content = f.readlines()
            self.permutation = np.random.permutation(range(0, len(content)))
            with open("results/"+file_md5_hash+"/permutation.txt", "a") as f:
                for value in self.permutation:
                    f.write(str(value)+"\n")
        else:
            # si existe, abro el archivo de permutacion y lo leo
            print("loading permutation folder!")
            with open("results/"+file_md5_hash+"/permutation.txt", "r") as f:
                self.permutation = []
                content = f.readlines()
                for value in content:
                    self.permutation.append(value)

    def speculative_execution(self, experimental_design, instances):
        self.initialize(experimental_design, instances)
        print("##############################")
        print("start_speculative_execution")
        while True:
            print("loop begin")
            probable_leaf = self._tree_descent()
            self._update_likelihood()
            best_alg = self._select_strategy2(probable_leaf)
            if best_alg is not None:
                self._execute(best_alg)
                self._save_results(probable_leaf)
            else:
                break
            print("loop end")
        print("###end_speculative_execution######################")

    def _tree_descent(self):
        """ Desciende por el arbol seleccionando la rama de la
        estrategia con mas probabilidades de ganar.
        si no hay suficientes instances ejecutadas, se corren las
        instances necesarias y se continua descendiendo hasta
        llegar a un nodo hoja """
        print("######start_tree_descent########")
        self.__msg = []
        self._tree_descent_strategies = {}
        if self.tree.root is None:
            print("raiz no existe, creando")
            self.tree.set_root(self._retrieve_node())
        node = self.tree.root
        print("raiz:", node)
        while True:
            self._tree_descent_strategies[hash(node.alg1)] = node.alg1
            self._tree_descent_strategies[hash(node.alg2)] = node.alg2
            if node.compare_strategies(self.pifile, self.instances, self.cpu_count):
                self.__msg.append(0)
                node.add_left(self._retrieve_node())
                if node.left.is_leaf == True:
                    self.tree_descent_outcome = node.left
                    node.msg = self.__msg
                    return node.left
                node = node.left
            else:
                self.__msg.append(1)
                node.add_right(self._retrieve_node())
                if node.right.is_leaf == True:
                    self.tree_descent_outcome = node.right
                    node.msg = self.__msg
                    return node.right
                node = node.right
            

        print("######end_tree_descent########")
        print("node at the end of tree_descent:", node)


    def _retrieve_node(self):
        """recibe como argumento un mensaje que contiene indicaciones
        para atravezar el arbol desde la raiz hasta ese nodo,
        y luego genera un nuevo nodo que consiste en un par de
        estrategias a comparar. para generar este nuevo nodo
        ejecuta el diseño experimental siguiendo las indicaciones del
        nodo. si ya hay un nodo similar en el arbol, ese nodo es
        retornado"""
        print("####start_retrieve_node")
        try:
            # print("_retrieve_node_try")
            self.__count = 0
            self.__speculativeNode = None
            self.experimental_design()
        except ValueError as x:
            print("_retrieve_node_except")
            print(self.__speculativeNode)
        else:
            print("_retrieve_node_else")
            self.marcarNodo()
        #finally:
        print("_retrieve_node_finally")
        print("nodo especulativo despues del try except")
        print(self.__speculativeNode)
        print("####end_retrieve_node")
        return self.__speculativeNode

    def marcarNodo(self):
        aux = self.tree.root
        self.__msg.pop()
        for i in self.__msg:
            if i == 0:
                aux = aux.left
                continue
            if i == 1:
                aux = aux.right
                continue
        aux.is_not_leaf = False
        msg_1 = self.__msg
        msg_2 = self.__msg
        msg_1.append(0)
        msg_2.append(1)
        aux.leaf_node_izq = LeafNode(msg_1)
        aux.leaf_node_der = LeafNode(msg_2)

    def bestStrategy(self, S1, S2):  # __range, delta_sig
        """
        Retorna la mejor estrategia entre dos,
        mediante el mensage especulativo.
        Si se alcanza el final del mensaje
        Setea el nodo especulativo correspondiente al estado del experimento
        lo crea si no existe
        y vuelve a retrieve node
        """
        print("comparing strategies")
        print (self.__count, self.__msg)
        if self.__count < len(self.__msg):
            if self.__msg[self.__count] == 0:
                self.__count = self.__count+1
                return S1
            else:
                self.__count = self.__count+1
                return S2
        experiment_state = TraceBackInfo.getExperimentState()
        if experiment_state in self.node_dict:
            self.__speculativeNode = self.node_dict[experiment_state]
            print(1,self.__speculativeNode)
        else:
            self.__speculativeNode = Node(S1, S2, len(self.instances), 0)
            self.node_dict[experiment_state] = self.__speculativeNode
            print(2,self.__speculativeNode)

        raise ValueError

    def _update_likelihood(self):
        print("#############################################################")
        print("diccionario de estrategias:", Strategy.strategy_instance_dict)
        print("#############################################################")
 
 
        print("passing info")
        self.sampler.pass_info(self.tree,Strategy.strategy_instance_dict,self.instances)


        for k in Strategy.strategy_instance_dict:
            alg = Strategy.strategy_instance_dict[k]
            if alg.needs_to_be_sampled:
               print("sampling summary:")
               data = alg.result_list()
               alg.sampledParameters = Sampler.sampleParameters(data)
               alg.tmpParameters = alg.sampledParameters 
               self.opt_res[alg], self.pes_res[alg] = self.sampler.sampledSum(alg, 95, 5, 1)
               
               data.append(self.opt_res[alg])
               alg.optimisticParameters = Sampler.sampleParameters(data)
               data.pop()
               
               data.append(self.pes_res[alg])
               alg.pessimisticParameters = Sampler.sampleParameters(data)
               data.pop()
               alg.needs_to_be_sampled = False   
               #print(data,opt_res,pes_res)                          



    def _execute(self, alg):
        """ dado un algoritmo ejecuta cierto numero de
        instancias y guarda los resultados en la matris de
        resultados globales"""
        print("runing algoritmo:"+str(alg))
        #  global self.instances,self.__numOfExecutions,self.pifile

        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        jobs = []
        for i in range(0, self.cpu_count):
            instance_index = alg.selectInstance()
            if instance_index >= len(self.instances):
                alg.isCompleted = True
                break
            instance = self.instances[instance_index]
            p = multiprocessing.Process(target=alg.run2, args=(instance, instance_index, self.pifile, return_dict))
            jobs.append(p)
        for p in jobs:
            p.start()
        for p in jobs:
            self.execution_num = self.execution_num + 1
            p.join()
        keys = [key for key, value in return_dict.items()]
        for k in keys:
            alg.addResult(k, return_dict[k])
        alg.needs_to_be_sampled = True

    def _save_results(self, max_leaf):
        with open("results/"+self.experiment_hash+"/results.txt", "a+") as res:
            execution_num = self.execution_num
            max_likelihood = self.max_sim_likelihood
            tree_desc_likelihood = self.tree_desc_likelihood
            print(str(execution_num) + "," + str(self.likelihood) + "," +str(self.max_like) + "," + str(self.min_like) +"," + self.output)
            res.write(str(execution_num) + "," + str(self.likelihood) + "," +str(self.max_like) + "," + str(self.min_like) +"," + self.output)
            for k in Strategy.strategy_instance_dict:
              alg = Strategy.strategy_instance_dict[k]  
              res.write( str(alg.lastInstanceIndex) + " ")
            res.write("\n")
            # res.write("{execution_num},{max_likelihood},{tree_desc_likelihood},{best_alg}\n")
        #a=input("Terminar?")
        #if a=="s": sys.exit()

    def _select_strategy2(self, max_leaf):
      self.sampler.simulations(self.__totalSimulations)
      self.likelihood = max_leaf.likelihood(self.__totalSimulations)

      max_volatility = -0.1
      best_strategy = None
      self.max_like=0.0
      self.min_like=1000.0

      for k in self._tree_descent_strategies:
         alg = self._tree_descent_strategies[k]
         if alg.isCompleted: continue
         alg.tmpParameters = alg.optimisticParameters
         alg.results[999] = self.opt_res[alg]
         self.sampler.simulations(self.__totalSimulations)
         opt_likelihood = max_leaf.likelihood(self.__totalSimulations)

         alg.tmpParameters = alg.pessimisticParameters
         alg.results[999] = self.pes_res[alg]
         self.sampler.simulations(self.__totalSimulations)
         pes_likelihood = max_leaf.likelihood(self.__totalSimulations) 
         alg.tmpParameters = alg.sampledParameters
         del alg.results[999]

         print(self.likelihood,opt_likelihood,pes_likelihood)
         volatility = max(self.likelihood,opt_likelihood,pes_likelihood) - min(self.likelihood,opt_likelihood,pes_likelihood)
         if volatility > max_volatility:
           max_volatility = volatility
           best_strategy = alg

         if max(self.likelihood,opt_likelihood,pes_likelihood)>self.max_like:
           self.max_like = max(self.likelihood,opt_likelihood,pes_likelihood)

         if min(self.likelihood,opt_likelihood,pes_likelihood)<self.min_like:
           self.min_like = min(self.likelihood,opt_likelihood,pes_likelihood)     
      
      return best_strategy
       
    def _toogle_anytime():
        self.anytime = not self.anytime

    def currentQuality(self):
        """
        Calculate the current likelihood of the
        tree.
        """
        print("calculting current quality")
        print("node_dict values:", self.node_dict.values())
        curr_quality = 0
        for node in self.node_dict.values():
            print("node ", node)
            print("not node.is_not_leaf: ", not node.is_not_leaf)
            if not node.is_not_leaf:
                if node.p1 > curr_quality:
                    curr_quality = node.p1
                elif node.p2 > curr_quality:
                    curr_quality = node.p2
        print("curr_quality", curr_quality)
        ret = curr_quality/self.__totalSimulations
        print("curr_quality", ret)
        return ret

    def terminate(self):
        """
        llegado el final del experimento se crea un nodo hoja que tiene como
        identificador el estado final del experimento.
        """

        experiment_state = TraceBackInfo.getExperimentState()
        if experiment_state in self.node_dict:
            self.__speculativeNode = self.node_dict[experiment_state]
        else:
            self.__speculativeNode = LeafNode(self.__msg, experiment_state)
            self.node_dict[experiment_state] = self.__speculativeNode

        raise ValueError
                

    """ funciones para agregar mas adelante"""
    #!----------------------------------------------------
    
    
    # def best_param_value(self):
    #     pass

    #!----------------------------------------------------


    # def bestParam(S0, param, values, pi, delta):
    #     SList = []
    #     for p in values:
    #         #  ---------------- crea copia de la estrategia
    #         S = Strategy(original=S0, new_params={param: p})
    #         #  -----------------S.params[param]=p
    #         S.name = 'Algo-'+param+"="+str(p)
    #         SList.append(S)

    #     Sbest = S0
    #     for S in SList:
    #         Sbest = bestStrategy(Sbest, S, pi, delta)
    #         if Sbest is not S0 : delta = 0.0
    #     return Sbest

    #!----------------------------------------------------


    # def run():
    #     print("welcome")
    #     #  -------------- Generar orden de instances

    #     self.instances = readData(pifile)
    #     #  -------------- Ejecucion
    #     plot_proc = multiprocessing.Process(target=plotter_function, args=())
    #     plot_proc.start()
    #     speculativeExecute()
    #     plot_proc.join()

    #!----------------------------------------------------

    #!----------------------------------------------------
    # def _execute(self, strategies):
    #     """ejecuta la lista de estrategias y actualiza
    #      los resutados globales"""
    #     print("#######start_execute##########")
    #     # print(alg.toString())
    #     # global instancias,__numOfExecutions,pifile,algoritmos
    #     manager = multiprocessing.Manager()
    #     return_dict = manager.dict()
    #     jobs = []
    #     numproc = self.cpu_count

    #     for j in range(numproc):
    #         i = alg.selectInstance()
    #         print("selected instance to run:")
    #         print(i)
    #         if i==None:
    #             alg.isCompleted = True#algoritmos.pop(mapa(alg))
    #             break
    #         instancia = instancias[i]
    #         p = multiprocessing.Process(target=alg.run2, args=(instancia,i,pifile,return_dict))
    #         jobs.append(p)
            
    #     # for p in jobs:
    #     #     p.start()    
    #     # for p in jobs:
    #     #     p.join()
    #     #     __numOfExecutions = __numOfExecutions + 1
    #     # keys = [key for key,value in return_dict.items()]
    #     # #print(keys)
    #     # for k in keys:
    #     #     alg.addResult(k, return_dict[k])

    #     # print("resultados post correr algoritmo:")
    #     # print(alg.results)
    #     # return True
    #     print("#######end_execute##########")
    #     pass