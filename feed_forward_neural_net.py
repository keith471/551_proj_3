'''Feed-forward neural-net'''

import numpy as np
import math

class FeedForwardNeuralNet(object):
    def __init__(self):
        # TODO
        pass

    def fprop(self):
        # TODO
        pass

    def bprop(self):
        # TODO
        pass

    def initialize(self):
        '''initialize the network'''
        pass
        # TODO

################################################################################
# activation functions
################################################################################
def sigmoid(a):
    return 1 / (1 + math.exp(-a))

################################################################################
# network components
################################################################################

# nodes

class NNNode(object):

    def __init__(self, bias_edge, in_edges, out_edges, output_activation=sigmoid):
        '''takes the bias, sets of input and output edges, as well as the output activation function'''
        self.bias_edge
        self.in_edges = in_edges
        self.out_edges = out_edges
        self.w, self.x = self.get_w_and_x(in_edges)
        self.output_activation = output_activation

    def input_activation(self):
        return self.bias_edge.weight + np.dot(self.w, self.x)

    def get_w_and_x(self, in_edges):
        w_l = [e.weight for e in in_edges]
        x_l = [e.value for e in in_edges]
        return np.array(w_l), np.array(x_l)

    def get_parents(self):
        return [e.tail for e in self.in_edges]

    def get_children(self):
        return [e.head for e in self.out_edges]

# edges

class GraphEdge(object):
    def __init__(self, head, tail, weight):
        self.head = head
        self.tail = tail
        self.value = value

class BiasEdge(GraphEdge):
    def __init__(self, head, tail, weight):
        GraphEdge.__init__(self, head, tail, weight)

class NNEdge(GraphEdge):
    def __init__(self, head, tail, weight, value):
        GraphEdge.__init__(self, head, tail, weight)
        self.value = value
