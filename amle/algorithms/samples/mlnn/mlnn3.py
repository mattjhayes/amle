"""
The mlnn3 module provides a multi-layer neural network
algorithm

Based on:
* Tutorial 'How to build a multi-layered neural network in Python'
  by Milo Spencer-Harper
  See: https://medium.com/technology-invention-and-more/how-to-build-a-multi \
                            -layered-neural-network-in-python-53ec3d1d326a
"""

import copy

from numpy import exp, array, random, dot

class Algorithm(object):
    """
    An algorithm module for import by AMLE
    """
    def __init__(self, logger, parameters):
        """
        Initialise the algorithm
        """
        self.logger = logger
        #*** Retrieve parameters passed to us:
        self.input_variables = parameters['input_variables']
        self.input_neurons = parameters['input_neurons']
        self.seed = parameters['random_seed']

        #*** Seed the random number generator:
        if self.seed:
            random.seed(self.seed)

        #*** Create NeuralNetwork instance:
        self.neuralnet = NeuralNetwork(logger)

        # TBD: do this from passed parameters:
        #*** Add neural layers:
        self.neuralnet.add_layer(inputs=3, neurons=4, name="hidden_layer")
        self.neuralnet.add_layer(inputs=4, neurons=1, name="output_layer")

    def initialise(self):
        """
        Use this to re-initialise before re-training
        """
        #*** Reinitialise the neurons:
        #self.layer1 = NeuronLayer(self.logger, self.input_neurons, self.input_variables)
        #self.layer2 = NeuronLayer(self.logger, 1, self.input_neurons)
        #self.neural_network = NeuralNetwork(self.layer1, self.layer2)

    def train(self, datasets, parameters):
        """
        Train the multi-layer neural network with training data
        """
        #*** Retrieve parameters passed to us:
        dataset = parameters['dataset']
        partition = parameters['partition']
        iterations = parameters['iterations']

        #*** Get the dataset ready:
        training_dset = datasets[dataset]
        training_inputs = training_dset.inputs_array(partition=partition)
        training_outputs = training_dset.outputs_array(partition=partition)

        self.logger.debug("training_inputs=\n%s", training_inputs)
        self.logger.debug("training_outputs=\n%s", training_outputs)

        # Train the neural network using the training set.
        # Do it many times and make small adjustments each time.
        self.neuralnet.train(training_inputs, training_outputs, iterations)

        self.logger.debug("Trained weights:")
        self.neuralnet.weights()

    def test(self, datasets, parameters):
        """
        Ask the multi-layer neural network for an answer to a situation
        and check result accuracy
        """
        #*** Retrieve parameters passed to us:
        dataset = parameters['dataset']
        partition = parameters['partition']

        #*** Get the dataset ready:
        test_dset = datasets[dataset]
        test_inputs = test_dset.inputs_array(partition=partition)
        test_outputs = test_dset.outputs_array(partition=partition)

        self.logger.debug("test_inputs=\n%s", test_inputs)
        self.logger.debug("test_outputs=\n%s", test_outputs)

        #*** Run thinking tests:
        results = []
        for index, input_array in enumerate(test_inputs):
            self.logger.debug("input_array=%s", input_array)
            output = self.neuralnet.think(input_array)
            self.logger.debug("output=%s", output)
            self.logger.debug("correct output=%s", test_outputs[index])
            results.append({'computed': output[0], 'actual': test_outputs[index][0]})
        return results

class NeuralNetwork(object):
    """
    Represents a neural network
    """
    def __init__(self, logger):
        """
        Initialise the neural network
        """
        self.logger = logger
        #*** Holds classes representing neural layers
        self.layers = []
        #*** Name string for debug etc:
        self.name = ""

    def add_layer(self, inputs, neurons, name):
        """
        Add a layer
        Last configured layer is assumed to be output layer
        """
        self.layers.append(NeuralLayer(self.logger, inputs, neurons, name))

    def train(self, inputs, outputs, iterations):
        """
        Train the neural network, running a number of training
        iterations (epochs) and adjusting synaptic weights each time
        """
        for iteration in xrange(iterations):
            output_layer = 1
            #*** Pass the training input set through the neural network:
            layer_outputs = self.feed_forward(inputs)

            #*** Work backward through the layers adjusting weights:
            for index, layer in reversed(list(enumerate(self.layers))):
                if output_layer:
                    layer_error = outputs - layer_outputs[index]
                    output_layer = 0
                else:
                    layer_error = layer_delta.dot(prev_synaptic_weights.T)

                layer_delta = layer_error * sigmoid_derivative(layer_outputs[index])

                #*** Calculate weight adjustment for the layer:
                if index > 0:
                    layer_adjustment = layer_outputs[index - 1].T.dot(layer_delta)
                else:
                    layer_adjustment = inputs.T.dot(layer_delta)
                
                prev_synaptic_weights = copy.copy(layer.synaptic_weights)
                
                #*** Adjust the layer weights:
                layer.synaptic_weights += layer_adjustment

    def feed_forward(self, inputs):
        """
        Passed inputs and return outputs from each layer
        """
        results = []
        for layer in self.layers:
            outputs = sigmoid(dot(inputs, layer.synaptic_weights))
            results.append(outputs)
            inputs = outputs
        return results

    def think(self, inputs):
        """
        Passed inputs and return outputs
        """
        for layer in self.layers:
            outputs = sigmoid(dot(inputs, layer.synaptic_weights))
            inputs = outputs
        return outputs

    def weights(self):
        """
        Return the synaptic weights for each layer
        """
        results = []
        for layer in self.layers:
            results.append(layer.synaptic_weights)
        return results

class NeuralLayer(object):
    """
    Generates an array of random synaptic weights for a
    layer of the neural network
    """
    def __init__(self, logger, inputs, neurons, name):
        self.synaptic_weights = 2 * random.random((inputs, neurons)) - 1
        logger.debug("%s synaptic_weights=\n%s", name, self.synaptic_weights)
        self.name = name

#=========================== Supporting Functions =============================
def sigmoid(x):
    """
    The Sigmoid function, which describes an S shaped curve.
    We pass the weighted sum of the inputs through this function to
    normalise them between 0 and 1. Uses exp (e) from numpy.
    """
    return 1 / (1 + exp(-x))

def sigmoid_derivative(x):
    """
    The derivative of the Sigmoid function.
    This is the gradient of the Sigmoid curve.
    It indicates how confident we are about the existing weight.
    """
    return x * (1 - x)
