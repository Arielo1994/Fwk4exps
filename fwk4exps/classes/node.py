from .leaf_node import LeafNode
import statistics


class Node(object):
    def __init__(self, s1, s2, total_instances, delta_sig):
        self.left = None
        self.right = None
        self.alg1 = s1
        self.alg2 = s2
        self.total_instances = total_instances
        self.delta_sig = delta_sig
        self.is_not_leaf = True
        #self.leaf_node_izq = None
        #self.leaf_node_der = None
        self.is_leaf = False

    def compare_strategies(self, pifile, instances, cpu_count):
        """ Compara la evaluacion parcial de las medias de ambas
        estrategias en este nodo, retorna true si la primera
        estrategia es mayor y false en caso contrario, si no hay
        instancias corre el numero minimo de instancias."""
        # print("#####start_compare_strategies")
        if self.alg1.no_results():
            self.alg1.run_minimum(pifile, instances, cpu_count)
            # print("self.alg1.results:",self.alg1.results)
        if self.alg2.no_results():
            self.alg2.run_minimum(pifile, instances, cpu_count)
            # print("self.alg2.results:",self.alg2.results)
        arr1 = []
        arr2 = []
        # print("self.alg1.results.keys()",self.alg1.results.keys())
        # print("self.alg2.results.keys()",self.alg2.results.keys())
        for k in self.alg1.results.keys() & self.alg2.results.keys():
            # print("common key:", k)
            arr1.append(self.alg1.results[k])
            arr2.append(self.alg2.results[k])
        # print(arr1,arr2)
        if statistics.mean(self.alg1.results.values()) > statistics.mean(self.alg2.results.values()):
        #if statistics.mean(arr1) > statistics.mean(arr2):
            # print("#####end_compare_strategies")
            return True
        else:
            # print("#####end_compare_strategies")
            return False

    # def getMsg(self):
    #     """retorna arreglo con resultados
    #     especulados en iteracion anterior
    #     p1 > p2
    #     Ej: [0,1,1,0]
    #     """
    #     msg = []
    #     node = self

    #     if self.p1 > self.p2:
    #         msg.insert(0, 0)
    #     else:
    #         msg.insert(0, 1)

    #     while(node.parent is None):
    #         if node.parent.left == node:
    #             msg.insert(0, 0)
    #         if node.parent.right == node:
    #             msg.insert(0, 1)
    #         node = node.parent

    #     return msg

    def add_left(self, node):
        self.left = node

    def add_right(self, node):
        self.right = node

    def __str__(self):
        return str(self.alg1)+" v/s "+str(self.alg2)

    def refreshSimulations(self):
        if self is not None:
            self.simulationVisitCount = 0
            self.p1 = 0
            self.p2 = 0

            # Then recur on left child
            if self.left is not None:
                self.left.refreshSimulations()

            # Finally recur on right child
            if self.right is not None:
                self.right.refreshSimulations()

    def addSimulationVisit(self):
        self.simulationVisitCount = self.simulationVisitCount + 1
