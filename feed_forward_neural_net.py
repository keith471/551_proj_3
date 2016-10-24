'''Feed-forward neural-net'''

from __future__ import print_function

# Just keep it simple and treat it as a single hidden layer neural net

import numpy as np
import math

class FeedForwardNeuralNet(object):
    def __init__(self):
        # TODO
        pass

    def fprop(self, x):
        # somehow feed x into the input layer
        # for each Layer
        #   compute the output of all neurons in the Layer
        #   and set this output as the input to the next Layer
        # finally, compute the output of the last Layer and return this

        # feed x into the input layer
        # TODO

        # fire the neurons in each layer
        for layer in self.network.layers:
            layer.fire()

        # compute and return the network output
        # TODO not everything is in place for this function to work
        return self.network.get_output()

    def bprop(self):
        # TODO is this done? I think so
        # recalculate weights for edges from hidden layer to output
        self.network.output_layer.set_delta()
        self.network.hidden_layer.update_out_weights()

        # recalculate weights for edges from input layer to hidden layer
        self.network.hidden_layer.set_deltas()
        self.network.input_layer.update_out_weights()

    def initialize(self):
        '''initialize the network'''
        # TODO
        # basically, just need to set up the neurons and initialize the weights
        # set self.network to...

        # in general:
        #   create nodes
        #   create edges with nodes
        #   add edges to nodes

        # create the input layer


        # create the hidden layer

        # create the ouput layer

        # create the network

################################################################################
# activation functions
################################################################################

# input
def input(b, w, x):
    return b + np.dot(w, x)

# output
def sigmoid(a):
    return 1 / (1 + math.exp(-a))

################################################################################
# network components
################################################################################

# nodes

class GraphNode(object):
    def add_edges(self, in_edges, out_edges):
        self.in_edges = in_edges
        self.out_edges = out_edges
        self.parents = self.get_parents()
        self.children = self.get_children()

    def get_parents(self):
        return [e.tail for e in self.in_edges]

    def get_children(self):
        return [e.head for e in self.out_edges]

class Neuron(GraphNode):
    def __init__(self, output_activation_func=sigmoid):
        GraphNode.__init__(self)
        self.output_activation_func = output_activation_func

    # override
    def add_edges(self, in_edges, out_edges):
        GraphNode.add_edges(in_edges, out_edges)
        self.set_w_and_x()

    def set_w_and_x(self):
        '''compose the input weights and values as vectors'''
        w_l = [e.weight for e in self.in_edges]
        x_l = [e.value for e in self.in_edges]
        self.w = np.array(w_l)
        self.x = np.array(x_l)

    ### TODO need an input activation function here

    def output_activation(self):
        '''output value of the neuron'''
        return self.output_activation_func(self.input_activation())

    def fire(self):
        '''fire the synapse between this neuron and its children; i.e. push a value
        on its outgoing edges'''
        output = self.output_activation()
        for edge in self.out_edges:
            edge.update_value(output)

    def update_out_weights(self):
        '''a neuron should be able to update the weights of its output edges'''
        for edge in self.out_edges:
            edge.update_weight()

class InputNeuron(Neuron):
    def __init__(self, output_activation_func=sigmoid):
        Neuron.__init__(self, output_activation_func)

    # TODO Does input node take a bias term?
    def input_activation(self):
        return np.dot(self.w, self.x)

    # override
    def get_parents(self):
        return None

class HiddenNeuron(Neuron):

    def __init__(self, output_activation_func=sigmoid):
        '''takes the bias, sets of input and output edges, as well as the output activation function'''
        Neuron.__init__(self, output_activation_func)

    def add_bias_edge(self, bias_edge):
        # TODO
        # remove this if all nodes take a bias
        # perhaps add a method somewhere to add a bias edge
        self.bias_edge = bias_edge

    def input_activation(self):
        # TODO remove or change this according to bias
        return self.bias_edge.weight + np.dot(self.w, self.x)

    # unique to hidden and output neurons
    def set_delta(self):
        output = self.output_activation()
        self.delta = output * (1 - output) * self.out_edges[0].weight * self.out_edges[0].head.delta

class OutputNeuron(Neuron):

    def __init__(self, output_activation_func):
        Neuron.__init__(self, output_activation_func)

    def add_bias_edge(self, bias_edge):
        # TODO
        # remove this if all nodes take a bias
        # perhaps add a method somewhere to add a bias edge
        self.bias_edge = bias_edge

    def input_activation(self):
        # TODO remove or change according to bias
        return self.bias_edge.weight + np.dot(self.w, self.x)

    def set_delta(self):
        output = self.get_output()
        # TODO what is y?
        self.delta = output * (1 - output) * (y - output)

# edges

class GraphEdge(object):
    def __init__(self, head, tail, weight):
        self.head = head
        self.tail = tail
        self.value = value

class Synapse(GraphEdge):
    def __init__(self, head, tail, weight, value):
        GraphEdge.__init__(self, head, tail, weight)
        self.value = value

    def update_weight(self):
        # TODO figure out what alpha is
        self.weight = self.weight + alpha * self.head.delta * self.value

    def update_value(self, value):
        '''set with the output of self.tail'''
        self.value = value

class BiasEdge(Synapse):
    def __init__(self, head, tail, weight):
        Synapse.__init__(self, head, tail, weight, None)

class InputEdge(Synapse):
    def __init__(self, head, value):
        Synapse.__init__(self, head, None, 1.0, value)

class HiddenEdge(Synapse):
    def __init__(self, head, tail, weight, value):
        Synapse.__init__(self, head, tail, weight, value)

class OutputEdge(Synapse):
    def __init__(self, tail, value):
        Synapse.__init__(self, None, tail, None, value)

# layer

class GraphLayer(object):
    def __init__(self, neurons, prev, next):
        self.neurons = neurons
        self.prev = prev
        self.next = next

class NetworkLayer(GraphLayer):
    def __init__(self, neurons, prev, next):
        GraphLayer.__init__(self, neurons, prev, next)

    def fire(self):
        '''fire all the neurons in the layer'''
        for neuron in self.neurons:
            neuron.fire()

    def update_out_weights(self):
        '''update the weights on the output edges of all neurons in the layer'''
        for neuron in self.neurons:
            neuron.update_out_weights()

class InputLayer(NetworkLayer):
    def __init__(self, neurons, next):
        GraphLayer.__init__(self, neurons, None, next)

class HiddenLayer(NetworkLayer):
    def __init__(self, neurons, prev, next):
        GraphLayer.__init__(self, neurons, prev, next)

    def set_deltas(self):
        for neuron in self.neurons:
            neuron.set_delta()

class OutputLayer(NetworkLayer):
    def __init__(self, neurons, prev):
        GraphLayer.__init__(self, neurons, prev, None)

    def get_output(self):
        return self.neurons[0].get_output()

    def set_delta(self):
        self.neurons[0].set_delta()

# network

def smash(ll):
    return [item for sublist in ll for item in sublist]

class Network(object):
    def __init__(self, input_layer, hidden_layer, output_layer):
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.hidden_layer = hidden_layer
        layers = [input_layer, hidden_layer, output_layer]
        self.layers = smash(layers)

    def get_output(self):
        '''get the output of the network'''
        return self.output_layer.get_output()

    def get_output_correction(self):
        return self.output_layer.get_correction()
