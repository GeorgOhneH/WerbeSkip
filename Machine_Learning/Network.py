from mnist_loader import load_mnist
from layers import FullyConnectedLayer, Dropout, ReLU, Sigmoid, TanH
from functions.costs import QuadraticCost
from optimizers import SGD, SGDMomentum, AdaGrad, RMSprop, Adam
from utils import make_mini_batches, Plotter

from random import randint
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
        self.optimizer = None
        self.input_neurons = 0
        self.layers = []
        self.biases = None
        self.weights = None
        self.cost = None
        self.train_loss = []
        self.train_accuracy = []
        self.validate_loss = []
        self.validate_accuracy = []
        self.costs = {
            "quadratic": QuadraticCost,
        }
        self.plotter = Plotter(self.train_loss, self.validate_loss, self.train_accuracy, self.validate_accuracy)

    def addInputLayer(self, neurons):
        self.input_neurons = neurons

    def addFullyConnectedLayer(self, neurons):
        self.layers.append(FullyConnectedLayer(neurons))

    def addActivation(self, activation):
        self.layers.append(activation)

    def addDropout(self, dropout):
        self.layers.append(Dropout(dropout))

    def regression(self, optimizer, cost="quadratic"):
        self.optimizer = optimizer
        self.cost = self.costs[cost]
        self.init()

    def init(self):
        # Uses Gaussian random variables with a mean of 0 and a standard deviation of 1
        neurons_before = self.input_neurons
        for layer in self.layers:
            neurons_before = layer.init(neurons_before, copy(self.optimizer))

    def feedforward(self, a):
        # Input is in Matrixform. Each row represents one datapoint
        for layer in self.layers:
            a = layer.forward(a)
        return a

    def fit(self, training_data_x, training_data_y, validation_data_x, validation_data_y, epochs, mini_batch_size,
            plot=False, snapshot_step=200):
        start_time = time.time()
        counter = 0
        for j in range(epochs):
            mini_batches = make_mini_batches(training_data_x, training_data_y, mini_batch_size)

            for index, mini_batch in enumerate(mini_batches):
                counter += 1
                self.update_weights(mini_batch)
                self.validate(validation_data_x, validation_data_y, mini_batch_size)

                if counter >= snapshot_step:
                    counter = 0
                    print(
                        "Epoch {} of {} | train_loss: {:.5f} | train_accuracy: {:.5f} | time {:.3f}\n"
                        "progress: {:.5f} | validation_loss: {:.5f} | validation_accuracy: {:.5f}".format(
                            j + 1, epochs, np.mean(self.train_loss[-100:]), np.mean(self.train_accuracy[-100:]), time.time() - start_time,
                            index / len(mini_batches), np.mean(self.validate_loss[-100:]), np.mean(self.validate_accuracy[-100:])),
                        sep='', end='\n', flush=True)

        if plot:
            self.plotter.plot_accuracy(epochs)
            self.plotter.plot_loss(epochs)

    def update_weights(self, mini_batch):
        x, y = mini_batch
        self.backprop(x, y)
        # Uses the error and adjusts the weights for each layer
        for layer in self.layers:
            layer.adjust_weights()

    def backprop(self, activation, y):
        for layer in self.layers:
            activation = layer.forward_backpropagation(activation)
        # https://sudeepraja.github.io/Neural/
        loss = self.cost.function(activation, y)
        self.train_loss.append(loss)

        accuracy = self.accuracy(activation, y)
        self.train_accuracy.append(accuracy)

        # calculates delta and saves it in each layer
        delta = self.cost.delta(activation, y)
        for layer in reversed(self.layers):
            delta = layer.make_delta(delta)

    def validate(self, x, y, size=None):
        if size is not None:
            rand = randint(0, x.shape[1]-size)
            x = x[..., rand:rand+size]
            y = y[..., rand:rand+size]

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

    def evaluate(self, x, y):
        orig_x = x
        wrong = []
        x = self.feedforward(x)
        loss = self.cost.function(x, y)
        if self.layers[-1].neurons != 2:
            accuracy = self.accuracy(x, y)
            print("Evaluation with {} data:\n"
                  "loss: {:.5f} | accuracy: {:.5f}".format(
                x.shape[1], loss, accuracy,
            ))
            return
        tp, tn, fp, fn = 0, 0, 0, 0  # True Positive, True Negative, False Positive, False Negative
        for index in range(x.shape[1]):
            a = np.argmax(x[:, index])
            b = np.argmax(y[:, index])
            if a == 1 and b == 1:
                tp += 1
            elif a == 0 and b == 0:
                tn += 1
            elif a == 1 and b == 0:
                fp += 1
                wrong.append((np.squeeze(np.asarray(orig_x[:, index])), "no_logo.BMP"))
            elif a == 0 and b == 1:
                fn += 1
                wrong.append((np.squeeze(np.asarray(orig_x[:, index])), "logo.BMP"))
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * (recall * precision) / (recall + precision)
        print("Evaluation with {} data:\n"
              "loss: {:.5f} | accuracy: {:.5f} | precision: {:.5f} | recall: {:.5f} | f1_score: {:.5f}".format(
            x.shape[1], loss, accuracy, precision, recall, f1_score
        ))
        return wrong


if __name__ == "__main__":
    train_data, train_labels, test_data, test_labels = load_mnist()
    net = Network()
    net.addInputLayer(28 * 28)
    net.addFullyConnectedLayer(100)
    net.addActivation(ReLU())
    net.addFullyConnectedLayer(100)
    net.addActivation(TanH())
    net.addFullyConnectedLayer(10)
    net.addActivation(Sigmoid())
    optimizer = Adam(learning_rate=0.001)
    net.regression(optimizer=optimizer, cost="quadratic")
    net.fit(train_data, train_labels, test_data, test_labels, epochs=20, mini_batch_size=20, plot=True)
    net.evaluate(test_data, test_labels)
    # best accuracy: 0.9822
