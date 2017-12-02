"""
The mlnn module provides a multi-layer neural network
algorithm

Based on excellent tutorial 'How to build a multi-layered neural network in Python'
by Milo Spencer-Harper

See: https://medium.com/technology-invention-and-more/how-to-build-a-multi \
                            -layered-neural-network-in-python-53ec3d1d326a
"""

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

        #*** Create neuron layer 1 (input layer):
        self.layer1 = NeuronLayer(logger, self.input_neurons, self.input_variables)

        #*** Create layer 2 with inputs equal to layer 1 neurons:
        self.layer2 = NeuronLayer(logger, 1, self.input_neurons)

        # Combine the layers to create a neural network
        self.neural_network = NeuralNetwork(logger, self.layer1, self.layer2)

    def initialise(self):
        """
        Use this to re-initialise before re-training
        """
        #*** Reinitialise the neurons:
        self.layer1 = NeuronLayer(self.logger, self.input_neurons, self.input_variables)
        self.layer2 = NeuronLayer(self.logger, 1, self.input_neurons)
        self.neural_network = NeuralNetwork(self.logger, self.layer1, self.layer2)

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
        self.neural_network.train(training_inputs, training_outputs, iterations)

        self.logger.debug("Trained weights:")
        self.logger.debug(" - Layer 1:\n%s", self.layer1.synaptic_weights)
        self.logger.debug(" - Layer 2:\n%s", self.layer2.synaptic_weights)

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
            hidden_state, output = self.neural_network.think(input_array)
            self.logger.debug("output=%s", output)
            self.logger.debug("correct output=%s", test_outputs[index])
            results.append({'computed': output[0], 'actual': test_outputs[index][0]})
        return results

class NeuronLayer():
    """
    Generates an array of random synaptic weights for a
    layer of the neural network
    """
    def __init__(self, logger, number_of_neurons, number_of_inputs_per_neuron):
        self.synaptic_weights = 2 * random.random((number_of_inputs_per_neuron,
                                                        number_of_neurons)) - 1
        logger.debug("synaptic_weights=\n%s", self.synaptic_weights)

class NeuralNetwork():
    """
    Represents a neural network
    """
    def __init__(self, logger, layer1, layer2):
        """
        Initialise the neural network
        """
        self.logger = logger
        #*** These will hold NeuronLayer instances:
        self.layer1 = layer1
        self.layer2 = layer2

    def __sigmoid(self, x):
        """
        The Sigmoid function, which describes an S shaped curve.
        We pass the weighted sum of the inputs through this function to
        normalise them between 0 and 1. Uses exp (e) from numpy.
        """
        return 1 / (1 + exp(-x))

    def __sigmoid_derivative(self, x):
        """
        The derivative of the Sigmoid function.
        This is the gradient of the Sigmoid curve.
        It indicates how confident we are about the existing weight.
        """
        return x * (1 - x)

    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        """
        Train the neural network, running a number of training
        iteration (epochs) and adjusting synaptic weights each time
        """
        for iteration in xrange(number_of_training_iterations):
            # Pass the training set through our neural network
            output_from_layer_1, output_from_layer_2 = self.think(training_set_inputs)

            # Calculate the error for layer 2 (The difference between the desired output
            # and the predicted output).
            layer2_error = training_set_outputs - output_from_layer_2
            layer2_delta = layer2_error * self.__sigmoid_derivative(output_from_layer_2)

            # Calculate the error for layer 1 (By looking at the weights in layer 1,
            # we can determine by how much layer 1 contributed to the error in layer 2).
            layer1_error = layer2_delta.dot(self.layer2.synaptic_weights.T)
            layer1_delta = layer1_error * self.__sigmoid_derivative(output_from_layer_1)

            # Calculate how much to adjust the weights by
            layer1_adjustment = training_set_inputs.T.dot(layer1_delta)
            layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)

            # Adjust the weights.
            self.layer1.synaptic_weights += layer1_adjustment
            self.layer2.synaptic_weights += layer2_adjustment

    def think(self, inputs):
        """
        Passed inputs and return outputs from each layer
        """
        output_from_layer1 = self.__sigmoid(dot(inputs, self.layer1.synaptic_weights))
        output_from_layer2 = self.__sigmoid(dot(output_from_layer1, self.layer2.synaptic_weights))
        return output_from_layer1, output_from_layer2

    def weights(self):
        """
        Return the synaptic weights for each layer
        """
        return self.layer1.synaptic_weights, self.layer2.synaptic_weights

