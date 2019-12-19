class LeafNode(object):
    def __init__(self, msg, state):
        # print("entra al init")
        self.output = None
        self.msg = msg
        self.simulations = 0
        self.state = state
        self.is_leaf = True
        # print("sale del init")

    def __str__(self):
        return str(self.msg)

    def add_simulation(self):
        self.simulations = self.simulations + 1

    def likelihood(self, total):
        return self.simulations/total

    def __eq__(self, other):
        return self.state == other.state
        
    def refreshSimulations(self):
        self.simulations = 0

    def addSimulationVisit(self):
        self.simulations = self.simulations + 1