class LeafNode(object):
    def __init__(self, msg, state):
        # print("entra al init")
        self.output = None
        self.msg = msg
        self.simulations = 0
        self.state = state
        # print("sale del init")

    def __str__(self):
        return str(self.msg)

    def add_simulation(self):
        self.simulations = self.simulations + 1

    def likelihood(self, total):
        return self.simulations/total

    def __eq__(self, other):
        return self.state == other.state
