# Import scipy.special for the sigmoid function expit()
import scipy.special, numpy

# Neural network class definition
class NeuralNetwork:
    # Init the network, this gets run whenever we make a new instance of this class
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set the number of nodes in each input, hidden and output layer
        self.i_nodes = input_nodes
        self.h_nodes = hidden_nodes
        self.o_nodes = output_nodes

        # Weight matrices, wih (input -> hidden) and who (hidden -> output)
        self.wih = numpy.random.normal(0.0, pow(self.h_nodes, -0.5), (self.h_nodes, self.i_nodes))
        self.who = numpy.random.normal(0.0, pow(self.o_nodes, -0.5), (self.o_nodes, self.h_nodes))

        # Set the learning rate
        self.lr = learning_rate

        # Set the activation function, the logistic sigmoid
        self.activation_function = lambda x: scipy.special.expit(x)

    # Train the network using back-propagation of errors
    def train(self, inputs_list, targets_list):
        # Convert inputs into 2D arrays
        inputs_array = numpy.array(inputs_list, ndmin=2).T
        targets_array = numpy.array(targets_list, ndmin=2).T

        # Calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs_array)

        # Calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # Calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)

        # Calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # Current error is (target - actual)
        output_errors = targets_array - final_outputs

        # Hidden layer errors are the output errors, split by the weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # Update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
        numpy.transpose(hidden_outputs))

        # Update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
        numpy.transpose(inputs_array))

    # Query the network
    def query(self, inputs_list):
        # Convert the inputs list into a 2D array
        inputs_array = numpy.array(inputs_list, ndmin=2).T

        # Calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs_array)

        # Calculate output from the hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # Calculate signals into final layer
        final_inputs = numpy.dot(self.who, hidden_outputs)

        # Calculate outputs from the final layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


# Create network with 3 inputs (1 is bias), 2 hidden neurones and 1 output

# Binary inputs
inputs_0 = [0.0, 0.0]
inputs_1 = [0.0, 1.0]
inputs_2 = [1.0, 0.0]
inputs_3 = [1.0, 1.0]
inputs = [inputs_0, inputs_1, inputs_2, inputs_3]

# Expected gate outputs from neural network
outputs_AND = [0, 0, 0, 1]
outputs_OR = [0, 1, 1, 1]
outputs_NAND = [1, 1, 1, 0]
outputs_NOR = [1, 0, 0, 0]
outputs_XOR = [0, 1, 1, 0]
outputs = [outputs_AND, outputs_OR, outputs_NAND, outputs_NOR, outputs_XOR]

# Bias needed to be added on to input array
bias = [-1.5, -0.5, 1.5, 0.5, 0] # doesn't matter what XOR bias is?

neurone = NeuralNetwork(3, 3, 1, 0.5)

# Loop for each gate
for y in range(5):
    # Loop for each input
    for x in range(len(inputs)):
        # Train
        for i in range(500):
            neurone.train(([bias[y]] + inputs[x]), outputs[y][x])
        print(neurone.query(([bias[y]] + inputs[x])))

