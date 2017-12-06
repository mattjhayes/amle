# -*- coding: utf-8 -*-
"""
The mlnn2 module provides a multi-layer neural network
algorithm

Based on excellent tutorial 'A Step by Step Backpropagation Example'
by Matt Mazur

See:
  https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

Heavily based on original code from:
  https://github.com/mattm/simple-neural-network/blob/master/neural-network.py
"""

import random
import math

from numpy import exp

#================== UNDER CONSTRUCTION, DOES NOT WORK IN AMLE YET =============

#
# Shorthand:
#   "pd_" as a variable prefix means "partial derivative"
#   "d_" as a variable prefix means "derivative"
#   "_wrt_" is shorthand for "with respect to"
#   "w_ho" and "w_ih" are the index of weights from hidden to output layer neurons and input to hidden layer neurons respectively
#
# Comment references:
#
# [1] Wikipedia article on Backpropagation
#   http://en.wikipedia.org/wiki/Backpropagation#Finding_the_derivative_of_the_error
# [2] Neural Networks for Machine Learning course on Coursera by Geoffrey Hinton
#   https://class.coursera.org/neuralnets-2012-001/lecture/39
#   (doesn't work anymore)
# [3] The Back Propagation Algorithm
#   https://www4.rgu.ac.uk/files/chapter3%20-%20bp.pdf
#   (doesn't work anymore)

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
        self.inputs = parameters['input_variables']
        self.hidden_neurons = parameters['input_neurons']
        self.seed = parameters['random_seed']
        
        # TBD, move to being a parameter:
        self.outputs = 1

        # TBD, remove static parameters:
        self.nn = NeuralNetwork(logger, self.inputs, self.hidden_neurons, self.outputs,
                    hidden_layer_weights=[0.15, 0.2, 0.25, 0.3],
                    hidden_layer_bias=0.35,
                    output_layer_weights=[0.4, 0.45, 0.5, 0.55],
                    output_layer_bias=0.6)

    def initialise(self):
        """
        Use this to re-initialise before re-training
        """
        # TBD, remove static parameters:
        self.nn = NeuralNetwork(self.logger, self.inputs, self.hidden_neurons, self.outputs,
                    hidden_layer_weights=[0.15, 0.2, 0.25, 0.3],
                    hidden_layer_bias=0.35,
                    output_layer_weights=[0.4, 0.45, 0.5, 0.55],
                    output_layer_bias=0.6)

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
        
        #*** Convert from numpy array to nested list:
        training_inputs = training_inputs.tolist()
        training_outputs = training_outputs.tolist()
        
        # TEMP: check type:
        print "type is ", type(training_inputs[0][0])
        
        self.logger.debug("training_inputs=\n%s", training_inputs)
        self.logger.debug("training_outputs=\n%s", training_outputs)

        # Train the neural network using the training set.
        # Do it many times and make small adjustments each time.
        for i in range(iterations):
            self.nn.train(training_inputs, training_outputs)

        #*** Display how neurons have been trained:
        self.logger.debug("Trained weights:")
        self.nn.inspect()

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
            output = self.nn.think(input_array)
            self.logger.debug("Thinking output=%s", output)
            self.logger.debug("correct output=%s", test_outputs[index])
            results.append({'computed': output[0], 'actual': test_outputs[index][0]})
        return results

class NeuralNetwork(object):
    LEARNING_RATE = 0.5

    def __init__(self, logger, num_inputs, num_hidden, num_outputs,
                    hidden_layer_weights = None, hidden_layer_bias = None,
                    output_layer_weights = None, output_layer_bias = None):
        """
        Initialise a NeuralNetwork
        """
        logger.info("NeuralNetwork initialising with num_inputs=%s, "
                    "num_hidden=%s, num_outputs=%s, hidden_layer_weights=%s",
                    num_inputs, num_hidden, num_outputs, hidden_layer_weights)
        logger.info("    hidden_layer_bias=%s, output_layer_weights=%s, "
                    "output_layer_bias=%s", hidden_layer_bias,
                     output_layer_weights, output_layer_bias)
   
        self.logger = logger
        self.num_inputs = num_inputs

        self.hidden_layer = NeuronLayer(logger, num_hidden, num_inputs, hidden_layer_bias)
        self.output_layer = NeuronLayer(logger, num_outputs, num_hidden, output_layer_bias)

        self.init_weights_from_inputs_to_hidden_layer_neurons(hidden_layer_weights)
        self.init_weights_from_hidden_layer_neurons_to_output_layer_neurons(output_layer_weights)

    def init_weights_from_inputs_to_hidden_layer_neurons(self, hidden_layer_weights):
        """
        Assign hidden layer weights, or if set to zero then assign
        randomly
        """
        self.logger.debug("init_weights_from_inputs_to_hidden_layer_neurons "
                               "hidden_layer_weights=%s", hidden_layer_weights)
        weight_num = 0
        for h in range(len(self.hidden_layer.neurons)):
            #self.logger.debug("h=%s", h)
            for i in range(self.num_inputs):
                #self.logger.debug("i=%s", i)
                if not hidden_layer_weights:
                    self.hidden_layer.neurons[h].weights.append(random.random())
                else:
                    #self.logger.debug("h=%s i=%s weight_num=%s", h, i, weight_num)
                    self.hidden_layer.neurons[h].weights.append(hidden_layer_weights[weight_num])
            weight_num += 1

    def init_weights_from_hidden_layer_neurons_to_output_layer_neurons(self, output_layer_weights):
        """
        TBD
        """
        self.logger.debug("init_weights_from_hidden_layer_neurons_to_output_layer_neurons "
                        "output_layer_weights=%s", output_layer_weights)
        weight_num = 0
        for o in range(len(self.output_layer.neurons)):
            for h in range(len(self.hidden_layer.neurons)):
                if not output_layer_weights:
                    self.output_layer.neurons[o].weights.append(random.random())
                else:
                    self.output_layer.neurons[o].weights.append(output_layer_weights[weight_num])
                weight_num += 1

    def inspect(self):
        """
        Display weightings and biases
        """
        print('------')
        print('* Inputs: {}'.format(self.num_inputs))
        print('------')
        print('Hidden Layer')
        self.hidden_layer.inspect()
        print('------')
        print('* Output Layer')
        self.output_layer.inspect()
        print('------')

    def feed_forward(self, inputs):
        """
        TBD
        """
        hidden_layer_outputs = self.hidden_layer.feed_forward(inputs)
        return self.output_layer.feed_forward(hidden_layer_outputs)

    def think(self, inputs):
        """
        Provide a set of inputs to the algorithm and return output(s)
        """
        self.logger.debug("NeuralNetwork thinking...")
        hidden_layer_outputs = self.hidden_layer.think_layer(inputs)
        self.logger.debug("NeuralNetwork hidden_layer_outputs=%s", hidden_layer_outputs)
        return self.output_layer.feed_forward(hidden_layer_outputs)

    # Uses online learning, ie updating the weights after each training case
    def train(self, training_inputs, training_outputs):
        """
        TBD
        """
        self.feed_forward(training_inputs)

        # 1. Output neuron deltas
        pd_errors_wrt_output_neuron_total_net_input = [0] * len(self.output_layer.neurons)
        for o in range(len(self.output_layer.neurons)):

            # ∂E/∂zⱼ
            pd_errors_wrt_output_neuron_total_net_input[o] = self.output_layer.neurons[o].calculate_pd_error_wrt_total_net_input(training_outputs[o])

        # 2. Hidden neuron deltas
        pd_errors_wrt_hidden_neuron_total_net_input = [0] * len(self.hidden_layer.neurons)
        for h in range(len(self.hidden_layer.neurons)):

            # We need to calculate the derivative of the error with respect to the output of each hidden layer neuron
            # dE/dyⱼ = Σ ∂E/∂zⱼ * ∂z/∂yⱼ = Σ ∂E/∂zⱼ * wᵢⱼ
            d_error_wrt_hidden_neuron_output = 0
            for o in range(len(self.output_layer.neurons)):
                d_error_wrt_hidden_neuron_output += pd_errors_wrt_output_neuron_total_net_input[o] * self.output_layer.neurons[o].weights[h]

            # ∂E/∂zⱼ = dE/dyⱼ * ∂zⱼ/∂
            pd_errors_wrt_hidden_neuron_total_net_input[h] = d_error_wrt_hidden_neuron_output * self.hidden_layer.neurons[h].calculate_pd_total_net_input_wrt_input()

        # 3. Update output neuron weights
        for o in range(len(self.output_layer.neurons)):
            for w_ho in range(len(self.output_layer.neurons[o].weights)):

                # ∂Eⱼ/∂wᵢⱼ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢⱼ
                pd_error_wrt_weight = pd_errors_wrt_output_neuron_total_net_input[o] * self.output_layer.neurons[o].calculate_pd_total_net_input_wrt_weight(w_ho)

                # Δw = α * ∂Eⱼ/∂wᵢ
                self.output_layer.neurons[o].weights[w_ho] -= self.LEARNING_RATE * pd_error_wrt_weight

        # 4. Update hidden neuron weights
        for h in range(len(self.hidden_layer.neurons)):
            for w_ih in range(len(self.hidden_layer.neurons[h].weights)):

                # ∂Eⱼ/∂wᵢ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢ
                pd_error_wrt_weight = pd_errors_wrt_hidden_neuron_total_net_input[h] * self.hidden_layer.neurons[h].calculate_pd_total_net_input_wrt_weight(w_ih)

                # Δw = α * ∂Eⱼ/∂wᵢ
                self.hidden_layer.neurons[h].weights[w_ih] -= self.LEARNING_RATE * pd_error_wrt_weight

    def calculate_total_error(self, training_sets):
        """
        TBD
        """
        total_error = 0
        for t in range(len(training_sets)):
            training_inputs, training_outputs = training_sets[t]
            self.feed_forward(training_inputs)
            for o in range(len(training_outputs)):
                total_error += self.output_layer.neurons[o].calculate_error(training_outputs[o])
        return total_error

class NeuronLayer(object):
    def __init__(self, logger, num_neurons, number_of_inputs, bias):
        """
        Initialise a neuron layer (multiple neurons)
        """
        logger.info("NeuronLayer initialising with num_neurons=%s "
                        "number_of_inputs=%s bias=%s", num_neurons,
                        number_of_inputs, bias)
        self.logger = logger
        # Every neuron in a layer shares the same bias
        self.bias = bias if bias else random.random()
        
        #** Every neuron in a layer is assumed to have same number of inputs
        self.number_of_inputs = number_of_inputs

        self.neurons = []
        for i in range(num_neurons):
            self.neurons.append(Neuron(logger, number_of_inputs, self.bias))

    def inspect(self):
        """
        TBD
        """
        print('Neurons:', len(self.neurons))
        for n in range(len(self.neurons)):
            print(' Neuron', n)
            for w in range(len(self.neurons[n].weights)):
                print('  Weight:', self.neurons[n].weights[w])
            print('  Bias:', self.bias)

    def feed_forward(self, inputs):
        """
        TBD
        """
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.calculate_output(inputs))
        return outputs

    def think_layer(self, inputs):
        """
        *** TEMP ***
        """
        outputs = []
        for neuron in self.neurons:
            output = neuron.calculate_output(inputs)
            self.logger.debug("NeuronLayer calculating neuron output=%s", output)
            outputs.append(output)
        self.logger.debug("outputs=%s", outputs)
        return outputs

    def get_outputs(self):
        """
        TBD
        """
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.output)
        return outputs

class Neuron(object):
    def __init__(self, logger, number_of_inputs, bias):
        """
        Initialise a neuron instance
        """
        self.logger = logger
        self.number_of_inputs = number_of_inputs
        self.bias = bias
        self.weights = []

    def calculate_output(self, inputs):
        """
        TBD
        """
        self.inputs = inputs
        self.output = self.squash(self.calculate_total_net_input())
        #self.output = self.__sigmoid(self.calculate_total_net_input())
        return self.output

    def calculate_total_net_input(self):
        """
        TBD
        """
        total = 0
        self.logger.debug("self.number_of_inputs=%s", self.number_of_inputs)
        for input_ in range(self.number_of_inputs):
            self.logger.debug("self.inputs[input_]=%s type=%s", self.inputs[input_], type(self.inputs[input_][0]))
            self.logger.debug("self.weights[input_]=%s type=%s", self.weights[input_], type(self.weights[input_]))
            total += self.inputs[input_] * self.weights[input_]
        return total + self.bias

    def squash(self, total_net_input):
        """
        Apply the logistic function to squash the output of the neuron
        The result is sometimes referred to as 'net' [2] or 'net' [1]
        """
        self.logger.debug("total_net_input=%s", total_net_input)
        return 1 / (1 + math.exp(-total_net_input))

    def __sigmoid(self, x):
        """
        The Sigmoid function, which describes an S shaped curve.
        We pass the weighted sum of the inputs through this function to
        normalise them between 0 and 1. Uses exp (e) from numpy.
        """
        return 1 / (1 + exp(-x))

    # Determine how much the neuron's total input has to change to move closer to the expected output
    #
    # Now that we have the partial derivative of the error with respect to the output (∂E/∂yⱼ) and
    # the derivative of the output with respect to the total net input (dyⱼ/dzⱼ) we can calculate
    # the partial derivative of the error with respect to the total net input.
    # This value is also known as the delta (δ) [1]
    # δ = ∂E/∂zⱼ = ∂E/∂yⱼ * dyⱼ/dzⱼ
    #
    def calculate_pd_error_wrt_total_net_input(self, target_output):
        """
        TBD
        """
        return self.calculate_pd_error_wrt_output(target_output) * self.calculate_pd_total_net_input_wrt_input();

    # The error for each neuron is calculated by the Mean Square Error method:
    def calculate_error(self, target_output):
        """
        TBD
        """
        return 0.5 * (target_output - self.output) ** 2

    # The partial derivate of the error with respect to actual output then is calculated by:
    # = 2 * 0.5 * (target output - actual output) ^ (2 - 1) * -1
    # = -(target output - actual output)
    #
    # The Wikipedia article on backpropagation [1] simplifies to the following, but most other learning material does not [2]
    # = actual output - target output
    #
    # Alternative, you can use (target - output), but then need to add it during backpropagation [3]
    #
    # Note that the actual output of the output neuron is often written as yⱼ and target output as tⱼ so:
    # = ∂E/∂yⱼ = -(tⱼ - yⱼ)
    def calculate_pd_error_wrt_output(self, target_output):
        """
        TBD
        """
        return -(target_output - self.output)

    # The total net input into the neuron is squashed using logistic function to calculate the neuron's output:
    # yⱼ = φ = 1 / (1 + e^(-zⱼ))
    # Note that where ⱼ represents the output of the neurons in whatever layer we're looking at and ᵢ represents the layer below it
    #
    # The derivative (not partial derivative since there is only one variable) of the output then is:
    # dyⱼ/dzⱼ = yⱼ * (1 - yⱼ)
    def calculate_pd_total_net_input_wrt_input(self):
        """
        TBD
        """
        return self.output * (1 - self.output)

    # The total net input is the weighted sum of all the inputs to the neuron and their respective weights:
    # = zⱼ = netⱼ = x₁w₁ + x₂w₂ ...
    #
    # The partial derivative of the total net input with respective to a given weight (with everything else held constant) then is:
    # = ∂zⱼ/∂wᵢ = some constant + 1 * xᵢw₁^(1-0) + some constant ... = xᵢ
    def calculate_pd_total_net_input_wrt_weight(self, index):
        """
        TBD
        """
        return self.inputs[index]

###

# Blog post example:

#nn = NeuralNetwork(2, 2, 2, hidden_layer_weights=[0.15, 0.2, 0.25, 0.3], hidden_layer_bias=0.35, output_layer_weights=[0.4, 0.45, 0.5, 0.55], output_layer_bias=0.6)
#for i in range(10000):
#    nn.train([0.05, 0.1], [0.01, 0.99])
#    print(i, round(nn.calculate_total_error([[[0.05, 0.1], [0.01, 0.99]]]), 9))

# XOR example:

# training_sets = [
#     [[0, 0], [0]],
#     [[0, 1], [1]],
#     [[1, 0], [1]],
#     [[1, 1], [0]]
# ]

# nn = NeuralNetwork(len(training_sets[0][0]), 5, len(training_sets[0][1]))
# for i in range(10000):
#     training_inputs, training_outputs = random.choice(training_sets)
#     nn.train(training_inputs, training_outputs)
#     print(i, nn.calculate_total_error(training_sets))
