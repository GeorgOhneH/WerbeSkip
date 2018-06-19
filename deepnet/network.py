from mnist_loader import load_mnist, load_conv
from layers import FullyConnectedLayer, Dropout, ReLU, BatchNorm, SoftMax, LReLU, Layer, ConvolutionLayer, MaxPoolLayer, Flatten
from functions.costs import QuadraticCost, CrossEntropyCost, Cost
from optimizers import Adam, Optimizer, SGD
from utils import make_mini_batches, Plotter, Analysis, Generator
from numpy import ndarray

import time
from copy import copy
import pickle
import os
import shutil

import numpy as np
from PIL import Image


class Network(object):
    """

    """

    def __init__(self):
        self._start_time = time.time()
        self._optimizer = None
        self._input_neurons = None
        self._cost = None
        self._layers = []
        self._train_loss = []
        self._train_accuracy = []
        self._validate_loss = []
        self._validate_accuracy = []
        self._costs = {
            "quadratic": QuadraticCost(),
            "cross_entropy": CrossEntropyCost()
        }
        self._translate = {
            "epoch": self._s_epoch,
            "progress": self._s_progress,
            "train_loss": self._s_tl,
            "train_accuracy": self._s_ta,
            "validate_loss": self._s_vl,
            "validate_accuracy": self._s_va,
            "time": self._s_time,
        }
        self._current_epoch = 0
        self._total_epoch = 0
        self._progress = 0
        self._plotter = Plotter(self)
        self._analysis = Analysis(self)

    @property
    def start_time(self) -> float:
        """read access only"""
        return self._start_time

    @property
    def cost(self) -> Cost:
        """read access only"""
        return self._cost

    @property
    def train_loss(self) -> list:
        """read access only"""
        return self._train_loss

    @property
    def train_accuracy(self) -> list:
        """read access only"""
        return self._train_accuracy

    @property
    def validate_loss(self) -> list:
        """read access only"""
        return self._validate_loss

    @property
    def validate_accuracy(self) -> list:
        """read access only"""
        return self._validate_accuracy

    def _s_epoch(self) -> str:
        return "epoch {} of {}".format(self._current_epoch+1, self._total_epoch)

    def _s_progress(self) -> str:
        return "progress: {:.3f}".format(self._progress)

    def _s_tl(self) -> str:
        return "train loss: {:.5f}".format(np.mean(self.train_loss[-100:]))

    def _s_ta(self) -> str:
        return "train accuracy: {:.5f}".format(np.mean(self.train_accuracy[-100:]))

    def _s_vl(self) -> str:
        return "validate loss: {:.5f}".format(np.mean(self.validate_loss[-100:]))

    def _s_va(self) -> str:
        return "validate accuracy: {:.5f}".format(np.mean(self.validate_accuracy[-100:]))

    def _s_time(self) -> str:
        return "time {:.3f}".format(time.time() - self.start_time)

    def input(self, neurons) -> None:
        """
        defines input size
        :param neurons: number of neurons
        :return: None
        """

        self._input_neurons = neurons

    def add(self, layer: Layer) -> None:
        """
        builds the network structure
        :param layer: Layer from the Layer class
        """
        if not issubclass(type(layer), Layer):
            raise ValueError("Must be a subclass of Layer not {}".format(type(layer)))

        self._layers.append(layer)

    def regression(self, optimizer: Optimizer, cost: str = "quadratic") -> None:
        """
        sets optimizer and cost
        init network
        :param optimizer: optimizer from the Optimizer class
        :param cost: string
        """
        if not issubclass(type(optimizer), Optimizer):
            raise ValueError("Must be a subclass of Optimizer not {}".format(type(optimizer)))

        if cost not in self._costs.keys():
            raise AssertionError("Must be one of these costs: {}".format(list(self._costs.keys())))

        self._optimizer = optimizer
        self._cost = self._costs[cost]
        self._init()

    def _init(self) -> None:
        """
        init all layers
        """
        if self._input_neurons is None:
            raise AttributeError("input() must be called first")

        neurons_before = self._input_neurons
        for layer in self._layers:
            neurons_before = layer.init(neurons_before, copy(self._optimizer))

    def fit(self,
            train_inputs: ndarray,
            train_labels: ndarray,
            validation_set: tuple or list = None,
            epochs: int = 10,
            mini_batch_size: int = 1,
            plot: bool = False,
            snapshot_step: int = 100,
            metrics: list = None) -> None:
        """
        tests if the given parameters are valid for the network
        :param train_inputs: Data
        :param train_labels: Rights result
        :param validation_set: Set to see the performance
        :param epochs: How often it goes through the train set
        :param mini_batch_size: Size of the mini_batch
        :param plot: plot the curve of the loss and the accuracy
        :param snapshot_step: use negative number to deactivate the monitoring
        :param metrics: define what metrics should be shown
        """
        if metrics is None:
            metrics = ["all"]
        if validation_set is not None:

            if not isinstance(validation_set, (tuple, list)):
                raise ValueError("Wrong type of validation set. Expected: {} not {}"
                                 .format((tuple, list), type(validation_set)))

            if len(validation_set) != 2:
                raise ValueError("Wrong length of validation set. Expected 2 not {}"
                                 .format(len(validation_set)))

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

        if not isinstance(metrics, list):
            raise ValueError("Wrong type for metrics. Expected {} not {}"
                             .format(list, type(metrics)))

        self._fit(
            train_inputs=train_inputs,
            train_labels=train_labels,
            validation_set=validation_set,
            epochs=epochs,
            mini_batch_size=mini_batch_size,
            plot=plot,
            snapshot_step=snapshot_step,
            metrics=metrics,
        )

    def _fit(self,
             train_inputs: ndarray,
             train_labels: ndarray,
             validation_set: set or list,
             epochs: int,
             mini_batch_size: int,
             plot: bool,
             snapshot_step: int,
             metrics: list) -> None:
        """
        trains the network with mini batches and print the progress.
        it can plot the accuracy and the loss
        """
        self._total_epoch = epochs
        for epoch in range(epochs):
            self._current_epoch = epoch
            mini_batches = make_mini_batches(train_inputs, train_labels, mini_batch_size)

            for index, mini_batch in enumerate(mini_batches):
                self._progress = index / len(mini_batches)
                self._update_parameters(mini_batch, mini_batch_size)

                if validation_set is not None:
                    self._analysis.validate(*validation_set, mini_batch_size)

                if metrics and len(self._train_loss) % snapshot_step == 0:
                    self._print_metrics(metrics)

        if plot:
            self._plot()

    def fit_generator(self,
                      generator: Generator,
                      validation_set: tuple or list = None,
                      plot: bool = False,
                      snapshot_step: int = 100,
                      metrics: list = None) -> None:
        """checks values"""
        # if not issubclass(type(generator), Generator):
        #     raise ValueError("Must be a subclass of Generator. "
        #                      "Use the Base Class from utils")

        if metrics is None:
            metrics = ["all"]
        self._fit_generator(generator=generator,
                            validation_set=validation_set,
                            plot=plot,
                            snapshot_step=snapshot_step,
                            metrics=metrics,
                            )

    def _fit_generator(self,
                       generator: Generator,
                       validation_set: tuple or list,
                       plot: bool,
                       snapshot_step: int,
                       metrics: list) -> None:
        """Same as fit but with a generator"""
        self._total_epoch = generator.epochs
        for epoch in range(generator.epochs):
            self._current_epoch = epoch
            for index, mini_batch in enumerate(generator):
                self._progress = index / len(generator)
                self._update_parameters(mini_batch, generator.mini_batch_size)

                if validation_set is not None:
                    self._analysis.validate(*validation_set, generator.mini_batch_size)
                if metrics and len(self._train_loss) % snapshot_step == 0:
                    self._print_metrics(metrics)

        if plot:
            self._plot()

    def _update_parameters(self, mini_batch: tuple or list, mini_batch_size: int) -> None:
        """
        After the backpropagation it adjust all layer's parameters
        """
        x, y = mini_batch

        self._backprop(x, y)

        for layer in self._layers:
            layer.adjust_parameters(mini_batch_size)

    def _backprop(self, x: ndarray, y: ndarray) -> None:
        """
        Computes the derivatives for each layer and
        saves the loss and accuracy
        :param x: data
        :param y: labels
        """
        for layer in self._layers:
            x = layer.forward_backpropagation(x)

        loss = self._cost.function(x, y)
        self._train_loss.append(loss)

        accuracy = self._analysis.accuracy(x, y)
        self._train_accuracy.append(accuracy)

        delta = self._cost.delta(x, y)
        for layer in reversed(self._layers):
            delta = layer.make_delta(delta)

    def _print_metrics(self, metrics: tuple or list) -> None:
        """
        prints all metric which are in the list.
        If "all" is in the first position it will
        print every metric.
        :param metrics: list with metrics as string
        """
        if metrics[0] == "all":
            metrics = self._translate.keys()

        result = ""
        for index, metric in enumerate(metrics):
            result += self._translate[metric]()
            if index + 1 < len(metrics):
                if len(result.split("\n")[-1]) > 60:
                    result += "\n"
                else:
                    result += " | "
        print(result)

    def _plot(self) -> None:
        """
        starts the plotting
        """
        self._plotter.plot_accuracy()
        self._plotter.plot_loss()

    def evaluate(self, x: ndarray, y: ndarray) -> None:
        """
        evaluates the network
        :param x: data
        :param y: labels
        """
        if self._cost is None:
            raise AssertionError("regression() must be called first")

        self._analysis.evaluate(x, y)

    def feedforward(self, a: ndarray) -> ndarray:
        """
        runs the input through the network and
        and returns the output
        :param a: data
        :return: processed data
        """
        for layer in self._layers:
            a = layer.forward(a)
        return a

    def predict(self, a: ndarray) -> ndarray:
        """
        runs the input through the network and
        returns the index of the max value
        :param a: data
        :return: 1-Dim list with the predictions
        """
        a = self.feedforward(a)
        return np.argmax(a, axis=1)

    def save(self, file_name: str) -> None:
        """
        saves the current network with all properties
        """
        with open("{}".format(file_name), "wb") as f:
            pickle.dump(self, f)

    def load(self, file_name: str) -> None:
        """
        Loads the network from a file
        """
        with open(file_name, "rb") as f:
            net = pickle.load(f)
        self.__dict__ = net.__dict__

    def print_network_structure(self) -> None:
        """
        Print the network structure
        Can be useful if you load a network from a file
        """
        print("Network Input: {}".format(self._input_neurons))
        for layer in self._layers:
            print("Layer: {}".format(layer))
        print("Cost: {}".format(self._cost))
        print("Optimizer: {}".format(self._optimizer))

    def save_wrong_predictions(self, inputs: ndarray, labels: ndarray, directory: str, shape: tuple or list) -> None:
        """
        WORKS ONLY FOR IMAGES!!!

        Saves all images that were wrongly predicted in the directory

        If the directory exist it will be overwritten and it will create subdirectories
        that are labeled with numbers corresponding to the output of the network
        this means if the network has 10 neurons as outputs. it will create 10 subdirectories
        labeled from 0 to 9

        The wrongly predicted images will be saved in those subdirectories

        With this schema:
            subdirectories: prediction from the network
            filename: <index>-<real label>-<output>

        :param inputs: data
        :param labels: labels
        :param directory: name of the directory
        :param shape: The shape of the Image
        """
        a = self.feedforward(inputs)

        # makes directories
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.mkdir(directory)
        [os.mkdir("{}\\{}".format(directory, x)) for x in range(labels.shape[1])]

        # saves images
        for index, (orig, result, label) in enumerate(zip(inputs.T, a.T, labels.T)):
            if np.argmax(result) != np.argmax(label):
                img_data = (orig * 255).reshape(shape).astype('uint8')
                Image.fromarray(img_data).save("{}\\{}\\{}-{}-{:.3f}.png"
                                               .format(directory, np.argmax(result), index, np.argmax(label),
                                                       np.max(result)))


if __name__ == "__main__":
    train_data, train_labels, test_data, test_labels = load_conv()
    net = Network()

    net.input((1, 28, 28))

    net.add(ConvolutionLayer(n_filter=32, width_filter=3,height_filter=3, stride=1, zero_padding=0))
    net.add(BatchNorm())
    net.add(ReLU())
    net.add(ConvolutionLayer(n_filter=64, width_filter=3,height_filter=3, stride=1, zero_padding=0))
    net.add(BatchNorm())
    net.add(ReLU())
    net.add(MaxPoolLayer(width_filter=2, height_filter=2, stride=1))
    net.add(Dropout(0.75))
    net.add(Flatten())
    net.add(FullyConnectedLayer(128))
    net.add(ReLU())
    net.add(Dropout(0.5))
    net.add(FullyConnectedLayer(10))
    net.add(SoftMax())

    optimizer = Adam(learning_rate=0.001)
    net.regression(optimizer=optimizer, cost="cross_entropy")

    net.fit(train_data, train_labels, validation_set=None, epochs=12, mini_batch_size=256,
            plot=True, snapshot_step=2)
    net.evaluate(test_data, test_labels)
