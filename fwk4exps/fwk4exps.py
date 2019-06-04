### imports
from fwk4exps.classes.plotter import plotter

### 

class Fwk4exps():

    def __init__(self, expDesign,PI):
        self.experimentalDesign = expDesign
        self.pi_path = PI
        self.__count = None
        self.__msg = None
        self.__speculativeNode = None
        self.root = None
        self.pifile=None
        self.experimentalDesign = None
        self.instancias = None
        self.global_results =None
        self.__totalSimulations = 100
        self.iteration = 0
        self.__numOfExecutions=0
        self.s2id = {}
        self.s_id =0
        self.the_end = False
        self.algoritmos = dict()
        self.bestAlg  = None
        self.hay_nodos_hoja = False
        self.node_state_dict = dict()
        self.quality_animation = None
        self.parameter_histogram =None
        self.quality_frame = None

    def run():
        print("welcome")
        self.instancias = readData(self.pi_path)
