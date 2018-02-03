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
        self.loss = np.array([])
        self.all_loss = []
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

    def fit(self, training_data_x, training_data_y, epochs, mini_batch_size, plot=False):
        for current_epoch, j in enumerate(range(epochs)):
            training_data_x, training_data_y = self.shuffle(training_data_x, training_data_y)
            mini_batches = [(training_data_x[:, k:mini_batch_size + k], training_data_y[:, k:mini_batch_size + k])
                            for k in range(0, training_data_y.shape[1], mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_weights(mini_batch, mini_batch_size)
                print("\rEpoch %d loss: " % (j + 1), self.loss[-1], sep='', end='', flush=True)
            print("\nEpoch %d out of %d is complete" % (j + 1, epochs))
            if plot:
                self.plot_loss(epochs, current_epoch+1)

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
        loss = self.layers[-1].calculate_loss(self.cost, y)
        self.loss = np.append(self.loss, loss)

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
        print("accuracy:", correct / n_data, "correct", correct," of ", n_data)

    def plot_loss(self, epochs, current_epoch):
        noisy_y_axis = self.loss[:]
        noisy_x_axis = np.arange(0, current_epoch, current_epoch / len(noisy_y_axis))

        window = int(len(noisy_y_axis) * 0.1)
        if window % 2 == 0:
            window -= 1
        print(len(noisy_y_axis))
        smooth_y_axis = savgol_filter(noisy_y_axis, window, 1)
        smooth_x_axis = np.arange(0, current_epoch, current_epoch / len(smooth_y_axis))

        plt.semilogy(noisy_x_axis, noisy_y_axis, color="lightblue", linewidth=0.1)
        plt.semilogy(smooth_x_axis, smooth_y_axis, color="blue", linewidth=0.5)

        plt.axis([-0.2, epochs * 1.05, -0.005, np.max(noisy_y_axis) * 1.2])
        plt.xlabel("Epochs")
        plt.ylabel("Loss")

        plt.show(block=True)

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
    net.fit(train_data, train_labels, epochs=4, mini_batch_size=10, plot=True)
    net.accuracy(test_data, test_labels)
    # best accuracy: 0.9822
