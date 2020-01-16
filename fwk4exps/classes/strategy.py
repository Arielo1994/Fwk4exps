import statistics
import copy
import subprocess
import random
import multiprocessing
import numpy as np
import pymc3 as pm
import os.path
import json
import hashlib


class Strategy(object):
    strategy_instance_dict = dict()
    permutation_folder = None

    def __new__(cls, *args, **kwargs):
        #print("__new__ method")
        instance = super(Strategy, cls).__new__(cls)
        if 'original' in kwargs:
            #print("copying strategy")
            original = kwargs['original']
            pathExe = original.pathExe
            __args = original.args
            if 'new_name' in kwargs:
                # print("new_name")
                name = kwargs['new_name']
            else:
                # print("no new_name")
                name = original.name

            params = original.params.copy()

            new_params = kwargs['new_params']
            for key in new_params:
                params[key] = new_params[key]

            strategy_string = pathExe + __args + name + str(tuple(params.values()))
            
            if strategy_string in cls.strategy_instance_dict:
                # print("there was already a copy")
                return cls.strategy_instance_dict[strategy_string]
            else:
                # print("creating copy instance")
                instance.name = name
                instance.pathExe = pathExe
                instance.args = __args
                instance.params = params

                instance.results = dict()
                #instance.sampledParameters = []
                #instance.optimisticParameters = []
                #instance.pessimisticParameters = []
                
                instance.simul_sums = []
                instance.optimistic_sums = []
                instance.pessimistic_sums = []
                instance.tmp_sums = []
                
                instance.lastInstanceIndex = -1
                instance.total = None
                instance.range = None
                instance.isCompleted = False
                instance.needs_to_be_sampled = False

                cls.strategy_instance_dict[strategy_string] = instance
        else:
            name = args[0]
            pathExe = args[1]
            __args = args[2]
            params = args[3]
            # print("name {0} pathexe {1} args {2} params {3}".format(name, pathExe, __args, params))
            strategy_string = pathExe + __args + name + str(tuple(params.values()))
            # print(strategy_string)

            if strategy_string in cls.strategy_instance_dict:
                # print("strategy found")
                return cls.strategy_instance_dict[strategy_string]
            else:
                # print("creating strategy")
                #input()
                instance.name = name
                instance.pathExe = pathExe
                instance.args = __args
                instance.params = params
                instance.results = dict()
                #instance.sampledParameters = []
                #instance.optimisticParameters = []
                #instance.pessimisticParameters = []

                instance.simul_sums = []
                instance.optimistic_sums = []
                instance.pessimistic_sums = []
                instance.tmp_sums = []

                instance.lastInstanceIndex = -1
                instance.total = None
                instance.range = None
                instance.isCompleted = False
                instance.needs_to_be_sampled = False
                instance.load_global_results()
                cls.strategy_instance_dict[strategy_string] = instance
                print("strategy dict: ")
                print(cls.strategy_instance_dict)

        return instance

    #def __init__(self, name=None, pathExe=None, args=None, params=None, original=None, new_params=None, new_name=None):
        # print("__init__ method")
        #print("succesfully created Strategy")

    def to_string(self):
        pass

    def no_results(self):
        return len(self.results) == 0

    def partial_mean(self):
        return statistics.mean(self.results.values())

    def selectInstance(self):
        self.lastInstanceIndex = self.lastInstanceIndex + 1
        return self.lastInstanceIndex

    def __str__(self):
        return self.name+" "+self.args.format(**self.params)

    def run(self, instance, i, PI):
        # PI = '/home/investigador/Documentos/algoritmo100real/Metasolver/extras/fw4exps/instancesBR.txt'
        # print("run function")
        # print("instance: {}, i: {}, PI: {}".format(instance,i,PI))
        aux = copy.copy(PI)
        # print("aux:", aux)
        aux = aux.split('/')
        # print("aux", aux)
        aux.pop()
        if len(aux)>0 :
          aux.pop(0)
          PI = ""
          for e in aux:
              PI = PI+"/"+e
          PI = PI+"/"
          instance = PI+instance
        # args = self.args
        # for k, v in self.params.items():
        #     args = args.replace(k, str(v))
        args = self.args.format(**self.params)

        commando = self.pathExe + " " + instance.rstrip() + " " + args
        print("comando:", commando)
        output = subprocess.getoutput(commando)
        output = output.splitlines()
        self.results[i] = float(output[-1])
        print("resultado: " + output[-1])
        # print("self.rsults:", self.results)
        return float(output[-1])

    def run2(self, instance, i, PI, return_dict):
        # PI = '/home/investigador/Documentos/algoritmo100real/Metasolver/extras/fw4exps/instancesBR.txt'
        return_dict[i] = self.run(instance, i, PI)

    def addResult(self, index, value):
        self.results[index] = value

    def run_minimum(self, pifile, instances, cpu_count):
        """ corre el minimo de instancias
        para poder hacer un calculo de estimacion de
        media (3)"""
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        jobs = []
        for i in range(0, cpu_count):
            instance_index = self.selectInstance()
            instance = instances[instance_index]
            p = multiprocessing.Process(target=self.run2, args=(instance, instance_index, pifile, return_dict))
            jobs.append(p)
        for p in jobs:
            p.start()    
        for p in jobs:
            p.join()
        keys = [key for key, value in return_dict.items()]
        for k in keys:
            self.addResult(k, return_dict[k])
        self.needs_to_be_sampled = True
        # print("resultados corrida minima:", self.results)

    def __hash__(self):
        params = tuple(self.params.values())
        return hash((self.pathExe, self.args, self.name)+params)

    def __eq__(self, other):
        return self.pathExe == other.pathExe and self.args == other.args and self.params == other.params and self.name == other.name

    def md5_hash(self):
        params = tuple(self.params.values())
        # return hash((self.pathExe, self.args, self.name)+params)
        data = tuple((self.pathExe, self.args, self.name))
        data = data + params
        data_json_string = json.dumps(data)
        return hashlib.md5(data_json_string.encode("utf-8")).hexdigest()    


    def addResult(self, index, value):
        self.results[index] = value
        with open("results/{}/strategies/{}.txt".format(self.permutation_folder, self.md5_hash()), "a") as f:
           f.write("{},{}\n".format(index, value))

    def result_list(self):
        return list(self.results.values())

    def to_file(self):
        pass

    def load_global_results(self):
        # revisar si existe archivo de esta estrategia
        # print("load_global_results")
        # print("permutation folder:", self.permutation_folder)
        path = "results/{}/strategies/{}.txt".format(self.permutation_folder, self.md5_hash())
        # print("path:", path)
        if os.path.isfile(path):
            # print("bbbbb")
            # si existe, leer linea por linea y añadir resultado
            with open(path) as f:
                content = f.readlines()
            for line in content:
                index, result = line.split(",")
                self.results[int(index)] = float(result)
            
            self.lastInstanceIndex = len(self.results)-1
            self.needs_to_be_sampled = True
        else:
            # si no existe, crear archivo
            # print("cccccc")
            with open("results/"+self.permutation_folder+"/strategies/strategy_dict.txt", "a") as f:
                f.write("{}:{}.txt\n".format(self, self.md5_hash()))
            with open(path, "w") as f:
                pass
            
