from mnist_loader import load_mnist
from layers import FullyConnectedLayer, Dropout, ReLU, Sigmoid, TanH, BatchNorm, SoftMax
from layers.layer import Layer
from functions.costs import QuadraticCost, CrossEntropyCost
from optimizers import SGD, SGDMomentum, AdaGrad, RMSprop, Adam
from optimizers.optimizer import Optimizer
from utils import make_mini_batches, Plotter, Analysis

import time
from copy import copy
import pickle
import os
import shutil

import numpy as np
from PIL import Image


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
            "cross_entropy": CrossEntropyCost
        }
        self._plotter = Plotter(self)
        self._analysis = Analysis(self)

    @property
    def cost(self):
        return self._cost

    @property
    def train_loss(self):
        return self._train_loss

    @property
    def train_accuracy(self):
        return self._train_accuracy

    @property
    def validate_loss(self):
        return self._validate_loss

    @property
    def validate_accuracy(self):
        return self._validate_accuracy

    def input(self, neurons):
        if not isinstance(neurons, int):
            raise ValueError("Must be an integer")

        self._input_neurons = neurons

    def add(self, layer):
        if not issubclass(type(layer), Layer):
            raise ValueError("Must be a subclass of Layer not {}".format(type(layer)))

        self._layers.append(layer)

    def regression(self, optimizer, cost="quadratic"):
        if not issubclass(type(optimizer), Optimizer):
            raise ValueError("Must be a subclass of Optimizer not {}".format(type(optimizer)))

        if cost not in self._costs.keys():
            raise AssertionError("Must be one of these costs: {}".format(list(self._costs.keys())))

        self._optimizer = optimizer
        self._cost = self._costs[cost]
        self._init()

    def _init(self):
        if self._input_neurons is None:
            raise AttributeError("input() must be called first")

        neurons_before = self._input_neurons
        for layer in self._layers:
            neurons_before = layer.init(neurons_before, copy(self._optimizer))

    def fit(self, train_input, train_labels, validation_set=None,
            epochs=10, mini_batch_size=1, plot=False, snapshot_step=200):

        if train_input.shape[0] != self._input_neurons:
            raise ValueError("Wrong shape of the train input. Expected ({}, x)"
                             .format(self._input_neurons))

        if validation_set is not None:

            if not isinstance(validation_set, (tuple, list)):
                return ValueError("Wrong type of validation set. Expected: {} not {}"
                                  .format((tuple, list), type(validation_set)))

            if len(validation_set) != 2:
                raise ValueError("Wrong length of validation set. Expected 2 not {}"
                                 .format(len(validation_set)))

            if validation_set[0].shape[0] != train_input.shape[0]:
                raise ValueError("Wrong shape of the validation input. Expected ({}, x)"
                                 .format(train_input.shape[0]))

        if not isinstance(epochs, int):
            raise ValueError("Wrong type for epoch. Expected {} not {}"
                             .format(int, type(epochs)))

        if not isinstance(mini_batch_size, int):
            raise ValueError("Wrong type for mini_batch_size. Expected {} not {}"
                             .format(int, type(mini_batch_size)))

        if not isinstance(plot, bool):
            raise ValueError("Wrong type for plot. Expected {} not {}"
                             .format(bool, type(plot)))

        if not isinstance(snapshot_step, int):
            raise ValueError("Wrong type for snapshot_step. Expected {} not {}"
                             .format(int, type(snapshot_step)))

        self._fit(
            train_input=train_input,
            train_labels=train_labels,
            validation_set=validation_set,
            epochs=epochs,
            mini_batch_size=mini_batch_size,
            plot=plot,
            snapshot_step=snapshot_step,
        )

    def _fit(self, train_input, train_labels, validation_set, epochs, mini_batch_size,
            plot, snapshot_step):
        start_time = time.time()
        counter = 0
        for j in range(epochs):
            mini_batches = make_mini_batches(train_input, train_labels, mini_batch_size)

            for index, mini_batch in enumerate(mini_batches):
                counter += 1
                self._update_parameters(mini_batch, mini_batch_size)
                self._analysis.validate(*validation_set, mini_batch_size)

                if counter >= snapshot_step:
                    counter = 0
                    print(
                        "Epoch {} of {} | train_loss: {:.5f} | train_accuracy: {:.5f} | time {:.3f}\n"
                        "progress: {:.5f} | validation_loss: {:.5f} | validation_accuracy: {:.5f}".format(
                            j + 1, epochs, np.mean(self._train_loss[-100:]), np.mean(self._train_accuracy[-100:]),
                            time.time() - start_time,
                            index / len(mini_batches), np.mean(self._validate_loss[-100:]),
                            np.mean(self._validate_accuracy[-100:])))

        if plot:
            self._plotter.plot_accuracy(epochs)
            self._plotter.plot_loss(epochs)

    def _update_parameters(self, mini_batch, mini_batch_size):
        x, y = mini_batch

        self._backprop(x, y)

        for layer in self._layers:
            layer.adjust_parameters(mini_batch_size)

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
        if self._cost is None:
            raise AssertionError("regression() must be called first")
        self._analysis.evaluate(x, y)

    def feedforward(self, a):
        for layer in self._layers:
            a = layer.forward(a)
        return a

    def predict(self, a):
        a = self.feedforward(a)
        return np.argmax(a, axis=0)

    def save(self, file_name):
        """
        saves the current network with all properties
        :param file_name:
        :return:
        """
        with open("{}.network".format(file_name), "wb") as f:
            pickle.dump(self, f)

    def load(self, file_name):
        """
        Loads the network from a file
        :param file_name: string
        :return: None
        """
        with open(file_name, "rb") as f:
            net = pickle.load(f)
        self.__dict__ = net.__dict__

    def save_wrong_predictions(self, inputs, labels, directory, shape):
        """
        WORKS ONLY FOR IMAGES!!!

        Saves all images that were wrongly predicted in the directory

        If the directory exist it will be overwritten and it will create subdirectories
        that are labeled with numbers corresponding to the output of the network
        this means if the network has 10 neurons as outputs. it will create 10 subdirectories
        labeled from 0 to 9

        The wrongly predicted images will be saved in those subdirectories

        With this schema:
            subdirectories = prediction from the network
            filename = <index>-<real label>

        :param inputs: ndarray
        :param labels: ndarray
        :param directory: string
        :param shape: tuple: (N, D)
        :return: None
        """
        a = self.feedforward(inputs)

        # makes directories
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.mkdir(directory)
        [os.mkdir("{}\\{}".format(directory, x)) for x in range(labels.shape[0])]

        # saves images
        for index, (orig, result, label) in enumerate(zip(inputs.T, a.T, labels.T)):
            if np.argmax(result) != np.argmax(label):
                img_data = np.multiply(orig, 255).reshape(shape).astype('uint8')
                Image.fromarray(img_data).save("{}\\{}\\{}-{}.png"
                                               .format(directory, np.argmax(result), index, np.argmax(label)))


if __name__ == "__main__":
    train_data, train_labels, test_data, test_labels = load_mnist()
    net = Network()

    net.input(28 * 28)

    net.add(FullyConnectedLayer(200))
    net.add(ReLU())

    net.add(FullyConnectedLayer(200))
    net.add(ReLU())

    net.add(FullyConnectedLayer(10))
    net.add(SoftMax())

    optimizer = Adam(learning_rate=0.01)
    net.regression(optimizer=optimizer, cost="cross_entropy")

    net.fit(train_data, train_labels, validation_set=(test_data, test_labels), epochs=1, mini_batch_size=128, plot=True)
    net.evaluate(test_data, test_labels)
    net.save_wrong_predictions(test_data, test_labels, directory="test", shape=(28, 28))
    net.save("hallo")
