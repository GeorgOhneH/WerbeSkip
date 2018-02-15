from mnist_loader import load_mnist
from layers import FullyConnectedLayer, Dropout, ReLU, Sigmoid, TanH, BatchNorm
from layers.layer import Layer
from functions.costs import QuadraticCost
from optimizers import SGD, SGDMomentum, AdaGrad, RMSprop, Adam
from optimizers.optimizer import Optimizer
from utils import make_mini_batches, Plotter, Analysis

import time
from copy import copy

import numpy as np


class Network(object):
    """
    Is a deep neural network which uses stochastic gradient descent
    and backpropagation to train the network
    It can use dynamically different activation-function and different cost-functions
    It supports only a FullyConnectedLayer
    The inputform is a numpymatrix, where the rows of the matrix the single dataelements represent
    """

    def __init__(self):
        self._optimizer = None
        self._input_neurons = None
        self._cost = None
        self._layers = []
        self._train_loss = []
        self._train_accuracy = []
        self._validate_loss = []
        self._validate_accuracy = []
        self._costs = {
            "quadratic": QuadraticCost,
        }
        self._plotter = Plotter(self._train_loss, self._validate_loss, self._train_accuracy, self._validate_accuracy)
        self._analysis = Analysis(self)

    @property
    def cost(self):
        return self._cost

    def input(self, neurons):
        if not isinstance(neurons, int):
            raise ValueError("Must be an integer")

        self._input_neurons = neurons

    def add(self, layer):
        if not issubclass(type(layer), Layer):
            raise ValueError("{} must be a subclass of Layer".format(layer))

        self._layers.append(layer)

    def regression(self, optimizer, cost="quadratic"):
        if not issubclass(type(optimizer), Optimizer):
            raise ValueError("{} must be a subclass of Optimizer".format(optimizer))

        if cost not in self._costs.keys():
            raise ValueError("{} must be one of these costs: {}".format(cost, list(self._costs.keys())))

        self._optimizer = optimizer
        self._cost = self._costs[cost]
        self._init()

    def _init(self):
        if self._input_neurons is None:
            raise AttributeError("input() must be called before regression()")

        neurons_before = self._input_neurons
        for layer in self._layers:
            neurons_before = layer.init(neurons_before, copy(self._optimizer))

    def feedforward(self, a):
        for layer in self._layers:
            a = layer.forward(a)
        return a

    def predict(self, a):
        a = self.feedforward(a)
        return np.argmax(a, axis=0)

    def fit(self, training_data_x, training_data_y, validation_data_x, validation_data_y, epochs, mini_batch_size,
            plot=False, snapshot_step=200):
        start_time = time.time()
        counter = 0
        for j in range(epochs):
            mini_batches = make_mini_batches(training_data_x, training_data_y, mini_batch_size)

            for index, mini_batch in enumerate(mini_batches):
                counter += 1
                self._update_parameters(mini_batch, mini_batch_size)
                self._analysis.validate(validation_data_x, validation_data_y, mini_batch_size)

                if counter >= snapshot_step:
                    counter = 0
                    print(
                        "Epoch {} of {} | train_loss: {:.5f} | train_accuracy: {:.5f} | time {:.3f}\n"
                        "progress: {:.5f} | validation_loss: {:.5f} | validation_accuracy: {:.5f}".format(
                            j + 1, epochs, np.mean(self._train_loss[-100:]), np.mean(self._train_accuracy[-100:]), time.time() - start_time,
                            index / len(mini_batches), np.mean(self._validate_loss[-100:]), np.mean(self._validate_accuracy[-100:])),
                        sep='', end='\n', flush=True)

        if plot:
            self._plotter.plot_accuracy(epochs)
            self._plotter.plot_loss(epochs)

    def _update_parameters(self, mini_batch, mini_batch_size):
        x, y = mini_batch

        self._backprop(x, y)

        for layer in self._layers:
            layer.adjust_weights(mini_batch_size)

    def _backprop(self, activation, y):
        for layer in self._layers:
            activation = layer.forward_backpropagation(activation)

        loss = self._cost.function(activation, y)
        self._train_loss.append(loss)

        accuracy = self._analysis.accuracy(activation, y)
        self._train_accuracy.append(accuracy)

        delta = self._cost.delta(activation, y)
        for layer in reversed(self._layers):
            delta = layer.make_delta(delta)

    def evaluate(self, x, y):
        self._analysis.evaluate(x, y)


if __name__ == "__main__":
    train_data, train_labels, test_data, test_labels = load_mnist()
    net = Network()

    net.input(28 * 28)

    net.add(FullyConnectedLayer(200))
    net.add(BatchNorm())
    net.add(ReLU())
    net.add(Dropout(0.8))

    net.add(FullyConnectedLayer(200))
    net.add(BatchNorm())
    net.add(ReLU())
    net.add(Dropout(0.8))

    net.add(FullyConnectedLayer(10))
    net.add(Sigmoid())

    optimizer = Adam(learning_rate=0.1)
    net.regression(optimizer=optimizer, cost="quadratic")
    net.fit(train_data, train_labels, test_data, test_labels, epochs=20, mini_batch_size=128, plot=True)
    net.evaluate(test_data, test_labels)
