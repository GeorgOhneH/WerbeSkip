from mnist_loader import load_mnist
from layers.fullyconnected import FullyConnectedLayer
from layers.input import InputLayer
from functions.activations import Sigmoid
from functions.costs import QuadraticCost

import numpy as np


class Network(object):
    def __init__(self):
        self.layers = []
        self.biases = None
        self.weights = None
        self.learning_rate = None
        self.cost = None
        self.loss = np.array([])
        self.all_loss = []
        self.activations = {
            "sigmoid": Sigmoid,
        }
        self.costs = {
            "quadratic": QuadraticCost,
        }

    def addInputLayer(self, neurons):
        self.layers.append(InputLayer(neurons))

    def addFullyConnectedLayer(self, neurons, activation="sigmoid"):
        self.layers.append(FullyConnectedLayer(neurons, self.activations[activation]))

    def regression(self, learning_rate=0.01, cost="quadratic"):
        self.learning_rate = learning_rate
        self.cost = self.costs[cost]
        self.init_weights()

    def init_weights(self):
        # Uses 
        for layer, layer_before in zip(self.layers[1:], self.layers[:-1]):
            layer.init_weights(layer_before.neurons)

    def feedforward(self, a):
        for layer in self.layers[1:]:
            a = layer.forward(a)
        return a

    def fit(self, training_data_x, training_data_y, epochs, mini_batch_size):
        # trains the neural network with  gradient descent and backpropagation
        for j in range(epochs):
            training_data_x, training_data_y = self.shuffle(training_data_x, training_data_y)
            mini_batches = [(training_data_x[:, k:mini_batch_size + k], training_data_y[:, k:mini_batch_size + k])
                            for k in range(0, training_data_y.shape[1], mini_batch_size)]
            for index, mini_batch in enumerate(mini_batches):
                self.update_weights(mini_batch, mini_batch_size)
                print("\rEpoch %d loss: " % (j + 1), np.sum(self.loss) / mini_batch_size / (index + 1), sep='', end='',
                      flush=True)
            # reset loss
            self.loss = np.array([])
            print()
            print("Epoch %d out of %d is complete" % (j + 1, epochs))

    def update_weights(self, mini_batch, mini_batch_size):
        for layer in self.layers[1:]:
            layer.reset_nabla()
        for index in range(mini_batch_size):
            x, y = mini_batch
            self.backprop(x[:, index], y[:, index])
        # Uses the error and adjusts the weights for each layer
        for layer in self.layers[1:]:
            layer.weights -= np.multiply(self.learning_rate / mini_batch_size, layer.nabla_w)
            layer.biases -= np.multiply(self.learning_rate / mini_batch_size, layer.nabla_b)

    def backprop(self, activation, y):
        # feedforward: supports multidimensional matrices
        for layer in self.layers[1:]:
            activation = layer.forward_backpropagation(activation)
        # backward pass: doesnt supports multidimensional matrices
        # https://sudeepraja.github.io/Neural/
        loss = self.layers[-1].calculate_loss(self.cost, y)
        self.loss = np.append(self.loss, [loss])

        # calculates delta and saves it in each layer
        delta = self.layers[-1].make_first_delta(self.cost, y)
        last_weights = self.layers[-1].weights
        for layer in reversed(self.layers[1:-1]):
            delta = layer.make_next_delta(delta, last_weights)
            last_weights = layer.weights

    def accuracy(self, x, y):
        n_data = x.shape[1]
        correct = 0
        x = self.feedforward(x)
        for index in range(n_data):
            a = x[:, index]
            if np.argmax(a) == np.argmax(y[:, index]):
                correct += 1
        print("accuracy: ", correct / n_data)

    # Input x = Matrix, y = Matrix
    def shuffle(self, x, y):
        # shuffles data in unison with helping from indexing
        indexes = np.random.permutation(x.shape[1])
        return x[:, indexes], y[:, indexes]


if __name__ == "__main__":
    train_data, train_labels, test_data, test_labels = load_mnist()
    net = Network()
    net.addInputLayer(28 * 28)
    net.addFullyConnectedLayer(50, activation="sigmoid")
    net.addFullyConnectedLayer(10, activation="sigmoid")
    net.regression(learning_rate=3, cost="quadratic")
    net.fit(train_data, train_labels, epochs=30, mini_batch_size=10)
    net.accuracy(test_data, test_labels)
    # best accuracy: 0.963
