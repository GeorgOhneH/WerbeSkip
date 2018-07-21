from deepnet.layers import Layer
from deepnet.functions.costs import QuadraticCost, CrossEntropyCost, Cost
from deepnet.optimizers import Optimizer
from deepnet.utils import make_mini_batches, Plotter, Analysis, Generator, IOHandler, make_batches

import time
from copy import copy
import os
import shutil
import importlib

import numpywrapper as np
from numpy import ndarray
import cv2
import deepdish as dd
from tqdm import tqdm


class Network(object):
    """

    """

    def __init__(self):
        self._use_gpu = False
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
        self.current_epoch = 0
        self.total_epoch = 0
        self.progress = 0
        self._plotter = Plotter(self)
        self._analysis = Analysis(self)
        self._iohandler = IOHandler(self)

    @property
    def use_gpu(self):
        return np.get_use_gpu()

    @use_gpu.setter
    def use_gpu(self, value: bool):
        np.set_use_gpu(value)
        importlib.reload(np)

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
            mini_batch_size: int = 128,
            plot: bool = False,
            snapshot_step: float = 1,
            metrics: list = None,
            save_step: None or int = None,
            path: str = "network.h5") -> None:
        """
        tests if the given parameters are valid for the network
        :param train_inputs: Data
        :param train_labels: Rights result
        :param validation_set: Set to see the performance
        :param epochs: How often it goes through the train set
        :param mini_batch_size: Size of the mini_batch
        :param plot: plot the curve of the loss and the accuracy
        :param snapshot_step: time (in seconds) between each print
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
            save_step=save_step,
            path=path
        )

    def _fit(self,
             train_inputs: ndarray,
             train_labels: ndarray,
             validation_set: set or list,
             epochs: int,
             mini_batch_size: int,
             plot: bool,
             snapshot_step: float,
             metrics: list,
             save_step: None or int,
             path) -> None:
        """
        trains the network with mini batches and print the progress.
        it can plot the accuracy and the loss
        """
        train_inputs = np.asarray(train_inputs)
        train_labels = np.asarray(train_labels)

        self.total_epoch = epochs
        for epoch in range(epochs):
            self.current_epoch = epoch
            mini_batches = make_mini_batches(train_inputs, train_labels, mini_batch_size)

            for index, mini_batch in enumerate(mini_batches):
                self.progress = index / len(mini_batches)
                self._update_parameters(mini_batch, mini_batch_size)

                if validation_set is not None:
                    self._analysis.validate(*validation_set, mini_batch_size)

                if save_step and index % save_step == 0:
                    self.save(path)

                self._iohandler.print_metrics(metrics, snapshot_step, mini_batch_size)

            if save_step:
                self.save(path)

        if plot:
            self._plot()

    def fit_generator(self,
                      generator: Generator,
                      validation_set: tuple or list = None,
                      plot: bool = False,
                      snapshot_step: float = 1,
                      metrics: list = None,
                      save_step: None or int = None,
                      path: str = "network.h5") -> None:
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
                            save_step=save_step,
                            path=path
                            )

    def _fit_generator(self,
                       generator: Generator,
                       validation_set: tuple or list,
                       plot: bool,
                       snapshot_step: float,
                       metrics: list,
                       save_step,
                       path) -> None:
        """Same as fit but with a generator"""
        self.total_epoch = generator.epochs
        for epoch in range(generator.epochs):
            self.current_epoch = epoch
            for index, mini_batch in enumerate(generator):
                self.progress = generator.progress / len(generator)
                self._update_parameters(mini_batch, generator.mini_batch_size)

                if validation_set is not None:
                    self._analysis.validate(*validation_set, generator.mini_batch_size)

                if save_step and index % save_step == 0:
                    self.save(path)

                self._iohandler.print_metrics(metrics, snapshot_step, generator.mini_batch_size)

            if save_step:
                self.save(path)

        if plot:
            self._plot()

    def _update_parameters(self, mini_batch: tuple or list, mini_batch_size: int) -> None:
        """
        After the backpropagation it adjust all layer's parameters
        """
        x, y = mini_batch

        self._backprop(np.asarray(x, dtype="float32"), np.asarray(y, dtype="float32"))

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
        self._train_loss.append(float(loss))

        accuracy = self._analysis.accuracy(x, y)
        self._train_accuracy.append(float(accuracy))

        delta = self._cost.delta(x, y)
        for layer in reversed(self._layers):
            delta = layer.make_delta(delta)

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
        batches = make_batches(a, 32)

        results = [self._feedforward(np.array(batch)) for batch in batches]
        out = np.concatenate(results)
        return out

    def _feedforward(self, a):
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

    def save(self, path: str) -> None:
        """
        saves the current network with all properties
        """
        meta = {
            "start_time": self._iohandler.start_time-time.time(),
            "inputs": self._iohandler.inputs,
            "train_loss": self._train_loss,
            "train_accuracy": self._train_accuracy,
            "validate_loss": self._validate_loss,
            "validate_accuracy": self._validate_accuracy,
        }
        parameters = [[np.asnumpy(value) for value in layer.save()] for layer in self._layers]
        dd.io.save(path, {"meta": meta, "parameters": parameters})

    def load(self, path: str) -> None:
        """
        Loads the network from a file
        """
        network = dd.io.load(path)

        parameters = network["parameters"]
        self._init()
        for layer, parameter in zip(self._layers, parameters):
            layer.load([np.asarray(value) for value in parameter])

        meta = network["meta"]
        self._iohandler.start_time = time.time() - meta["start_time"]
        self._iohandler.inputs = meta["inputs"]
        self._train_loss = meta["train_loss"]
        self._train_accuracy = meta["train_accuracy"]
        self._validate_loss = meta["validate_loss"]
        self._validate_accuracy = meta["validate_accuracy"]

    def print_network_structure(self) -> None:
        """
        Print the network structure
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
            filename: <index>-<real label>-<prediction>

        :param inputs: data
        :param labels: labels
        :param directory: name of the directory
        :param shape: The shape of the Image
        """
        inputs = np.asarray(inputs)
        labels = np.asarray(labels)

        a = self.feedforward(inputs)

        # makes directories
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.mkdir(directory)
        [os.mkdir("{}\\{}".format(directory, x)) for x in range(labels.shape[1])]

        # saves images
        for index, (orig, prediction, label) in enumerate(zip(inputs, a, labels)):
            if np.argmax(prediction) != np.argmax(label):
                img_data = np.asnumpy(orig * 255).reshape(shape).astype('uint8').transpose(1, 2, 0)
                file_name = "{}\\{}\\{}-{}-{:.3f}.png".format(directory, int(np.argmax(prediction)), int(index),
                                                              int(np.argmax(label)), float(np.amax(prediction)))
                cv2.imwrite(file_name, img_data)
