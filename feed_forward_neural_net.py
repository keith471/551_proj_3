'''Feed-forward neural-net'''

# TODO add InputNeuron class
# replace NNNeuron class with HiddenNeuron
# add InputLayer class
# replace NNLayer with HiddenLayer
# Just keep it simple and treat it as a single layer neural net

import numpy as np
import math

class FeedForwardNeuralNet(object):
    def __init__(self):
        # TODO
        pass

    def fprop(self, x):
        # TODO
        # somehow feed x into the input layer
        # for each Layer
        #   compute the output of all neurons in the Layer
        #   and set this output as the input to the next Layer
        # finally, compute the output of the last Layer and return this
        for layer in self.network.layers:
            for neuron in layer.neurons:
                neuron.fire()
        return self.network.get_output()

    def bprop(self):
        # TODO
        # assume this is called with the correction of the output nodes already set
        # for each hidden unit, compute its share of the correction
        # update each network weight

        # set delta on the output
        self.network.output_layer.set_delta()



        for layer in reversed(self.network.hidden_layers):
            for neuron in layer.neurons:

                neuron.update_out_weights()

    def initialize(self):
        '''initialize the network'''
        # basically, just need to set up the neurons and initialize the weights
        # set self.network to...
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

class NNNeuron(object):

    def __init__(self, bias_edge, in_edges, out_edges, output_activation_func=sigmoid):
        '''takes the bias, sets of input and output edges, as well as the output activation function'''
        self.bias_edge = bias_edge
        self.in_edges = in_edges
        self.out_edges = out_edges
        self.w, self.x = self.get_w_and_x(in_edges)
        self.output_activation_func = output_activation_func
        self.parents = self.get_parents()
        self.children = self.get_children()

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

    def output_activation(self):
        '''output value of the neuron'''
        return self.output_activation_func(self.input_activation())

    def fire(self):
        '''fire the synapse between this neuron and its children; i.e. push a value
        on its outgoing edges'''
        output = self.output_activation()
        for edge in self.out_edges:
            edge.set_value(output)

    def set_delta(self):
        output = self.output_activation()
        self.delta = output * (1 - output) *


    def update_out_weights(self):
        for edge in self.out_edges:
            # calculate delta

            edge.update_weight()

class OutputNeuron(object):

    def __init__(self, bias_edge, in_edges):
        self.bias_edge = bias_edge
        self.in_edges

    def get_output(self):
        '''returns an array of probabilities for each class, or perhaps just the
        class with highest probability'''
        pass
        # TODO

    def fire(self):
        '''compute the output of the neuron'''
        pass
        # TODO

    def set_delta(self):
        output = self.get_output()
        # what is y?
        self.delta = output * (1 - output) * (y - output)

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

    def set_value(self, value):
        '''set with the output of self.tail'''
        self.value = value

    def update_weight(self):
        # get delta

# layer

class GraphLayer(object):
    def __init__(self, neurons, prev, next):
        self.neurons = neurons
        self.prev = prev
        self.next = next

class NNLayer(GraphLayer):
    def __init__(self, neurons, prev, next):
        GraphLayer.__init__(self, neurons, prev, next)

class OutputLayer(GraphLayer):
    def __init__(self, neurons, prev):
        GraphLayer.__init__(self, neurons, prev, None)

    def get_output(self):
        return self.neurons[0].get_output()

    def set_deltas(self):
        for neuron in self.neurons:
            neuron.set_delta()
# network

def smash(ll):
    return [item for sublist in ll for item in sublist]

class NNNetwork(object):

    def __init__(self, input_layer, hidden_layers, output_layer):
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.hidden_layers = hidden_layers
        layers = [input_layer, hidden_layers, output_layer]
        self.layers = smash(layers)

    def get_output(self):
        '''get the output of the network'''
        return self.output_layer.get_output()

    def get_output_correction(self):
        return self.output_layer.get_correction()
