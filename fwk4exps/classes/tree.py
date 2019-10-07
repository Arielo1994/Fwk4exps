from .node import Node


class Tree(object):
    def __init__(self):
        self.root = None

    def set_root(self, root):
        print("set_root:", root)
        self.root = root

    def print_tree(self):
        pass

    def refreshSimulations(self):
        self.root.refreshSimulations()
