from mnist_loader import load_mnist
from layers.fullyconnected import FullyConnectedLayer
from layers.input import InputLayer
from functions.activations import Sigmoid, ReLU
from functions.costs import QuadraticCost

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from scipy.signal import savgol_filter


class Network(object):
    """
    Is a deep neural network which uses stochastic gradient descent
    and backpropagation to train the network
    It can use dynamically different activation-function and different cost-functions
    It supports only a FullyConnectedLayer
    The inputform is a numpymatrix, where the rows of the matrix the single dataelements represent
    """
    def __init__(self):
        self.layers = []
        self.biases = None
        self.weights = None
        self.learning_rate = None
        self.cost = None
        self.train_loss = []
        self.train_accuracy = []
        self.validate_loss = []
        self.validate_accuracy = []
        self.activations = {
            "sigmoid": Sigmoid,
            "relu": ReLU,
        }
        self.costs = {
            "quadratic": QuadraticCost,
        }

    def addInputLayer(self, neurons):
        self.layers.append(InputLayer(neurons))

    def addFullyConnectedLayer(self, neurons, activation="sigmoid", dropout=None):
        self.layers.append(FullyConnectedLayer(neurons, self.activations[activation], dropout))

    def regression(self, learning_rate=0.01, cost="quadratic"):
        self.learning_rate = learning_rate
        self.cost = self.costs[cost]
        self.init()

    def init(self):
        # Uses Gaussian random variables with a mean of 0 and a standard deviation of 1
        for layer, layer_before in zip(self.layers[1:], self.layers[:-1]):
            layer.init(layer_before.neurons)

    def feedforward(self, a):
        # Input is in Matrixform. Each row represents one datapoint
        for layer in self.layers[1:]:
            a = layer.forward(a)
        return a

    def fit(self, training_data_x, training_data_y, validation_data_x, validation_data_y, epochs, mini_batch_size, plot=False, monitoring=True):
        for j in range(epochs):
            training_data_x, training_data_y = self.shuffle(training_data_x, training_data_y)
            mini_batches = [(training_data_x[:, k:mini_batch_size + k], training_data_y[:, k:mini_batch_size + k])
                            for k in range(0, training_data_y.shape[1], mini_batch_size)]
            for index, mini_batch in enumerate(mini_batches):
                self.update_weights(mini_batch, mini_batch_size)

            if monitoring:
                validation_loss, validation_accuracy = self.validate(validation_data_x, validation_data_y)
                print(
                    "Epoch {} | train_loss: {:.5f} | train_accuracy: {:.5f}\n"
                    "validation_loss: {:.5f} | validation_accuracy: {:.5f}".format(
                        j+1, np.mean(self.train_loss[-len(mini_batches):]),
                        self.train_accuracy[-1],
                        validation_loss, validation_accuracy),
                    sep='', end='\n', flush=True)

        if plot:
            self.plot_loss(epochs)

    def update_weights(self, mini_batch, mini_batch_size):
        x, y = mini_batch
        self.backprop(x, y)
        # Uses the error and adjusts the weights for each layer
        for layer in self.layers[1:]:
            layer.weights -= np.multiply(self.learning_rate / mini_batch_size, layer.nabla_w)
            layer.biases -= np.multiply(self.learning_rate / mini_batch_size, layer.nabla_b)

    def backprop(self, activation, y):
        for layer in self.layers[1:]:
            activation = layer.forward_backpropagation(activation)
        # https://sudeepraja.github.io/Neural/
        loss = self.cost.function(activation, y)
        self.train_loss.append(loss)

        accuracy = self.accuracy(activation, y)
        self.train_accuracy.append(accuracy)

        # calculates delta and saves it in each layer
        delta = self.layers[-1].make_first_delta(self.cost, y)
        last_weights = self.layers[-1].weights
        for layer in reversed(self.layers[1:-1]):
            delta = layer.make_next_delta(delta, last_weights)
            last_weights = layer.weights

    def validate(self, x, y):
        x = self.feedforward(x)
        loss = self.cost.function(x, y)
        accuracy = self.accuracy(x, y)
        self.validate_accuracy.append(accuracy)
        self.validate_loss.append(loss)
        return loss, accuracy

    def accuracy(self, x, y):
        n_data = x.shape[1]
        correct = 0
        for index in range(n_data):
            a = x[:, index]
            if np.argmax(a) == np.argmax(y[:, index]):
                correct += 1
        return correct / n_data

    def plot_loss(self, epochs):
        noisy_y_axis = self.train_loss[:]

        window = int(len(noisy_y_axis) * 0.05)
        if window % 2 == 0:
            window -= 1

        smooth_y_axis = savgol_filter(noisy_y_axis, window, 1)
        smooth_x_axis = np.arange(0, epochs, epochs / len(smooth_y_axis))

        plt.semilogy(smooth_x_axis, smooth_y_axis, color="blue", linewidth=1, label="train")

        plt.axis([0, epochs, np.min(smooth_y_axis), np.max(smooth_y_axis)])

        plt.title("model loss")
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.legend()

        plt.ioff()
        plt.show()

    # Input x = Matrix, y = Matrix
    def shuffle(self, x, y):
        # shuffles data in unison with helping from indexing
        indexes = np.random.permutation(x.shape[1])
        return x[..., indexes], y[..., indexes]


if __name__ == "__main__":
    train_data, train_labels, test_data, test_labels = load_mnist()
    net = Network()
    net.addInputLayer(28 * 28)
    net.addFullyConnectedLayer(100, activation="relu", dropout=0.8)
    net.addFullyConnectedLayer(10, activation="sigmoid")
    net.regression(learning_rate=1, cost="quadratic")
    net.fit(train_data, train_labels, test_data, test_labels, epochs=4, mini_batch_size=10, plot=True, monitoring=True)
    print(net.accuracy(test_data, test_labels))
    # best accuracy: 0.9822
