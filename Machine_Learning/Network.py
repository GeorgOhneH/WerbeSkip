from mnist_loader import load_mnist
from layers import InputLayer, FullyConnectedLayer, Dropout
from functions.activations import Sigmoid, ReLU
from functions.costs import QuadraticCost

from random import randint
import time

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

    def addFullyConnectedLayer(self, neurons, activation="sigmoid"):
        self.layers.append(FullyConnectedLayer(neurons, self.activations[activation]))

    def addDropout(self, dropout):
        self.layers.append(Dropout(dropout))

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

    def fit(self, training_data_x, training_data_y, validation_data_x, validation_data_y, epochs, mini_batch_size,
            plot=False, snapshot_step=200):
        start_time = time.time()
        counter = 0
        for j in range(epochs):
            training_data_x, training_data_y = self.shuffle(training_data_x, training_data_y)
            mini_batches = [(training_data_x[:, k:mini_batch_size + k], training_data_y[:, k:mini_batch_size + k])
                            for k in range(0, training_data_y.shape[1], mini_batch_size)]

            for index, mini_batch in enumerate(mini_batches):
                counter += 1
                self.update_weights(mini_batch, mini_batch_size)
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
            self.plot_loss(epochs)
            self.plot_accuracy(epochs)

    def update_weights(self, mini_batch, mini_batch_size):
        x, y = mini_batch
        self.backprop(x, y)
        # Uses the error and adjusts the weights for each layer
        for layer in self.layers[1:]:
            layer.adjust_weights(self.learning_rate / mini_batch_size)

    def backprop(self, activation, y):
        for layer in self.layers[1:]:
            activation = layer.forward_backpropagation(activation)
        # https://sudeepraja.github.io/Neural/
        loss = self.cost.function(activation, y)
        self.train_loss.append(loss)

        accuracy = self.accuracy(activation, y)
        self.train_accuracy.append(accuracy)

        # calculates delta and saves it in each layer
        delta, last_weights = self.layers[-1].make_first_delta(self.cost, y)
        for layer in reversed(self.layers[1:-1]):
            delta, last_weights = layer.make_delta(delta, last_weights)

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

    def plot_loss(self, epochs):
        smooth_train_x_axis, smooth_train_y_axis = self.smooth_data(self.train_loss, epochs)
        smooth_validation_x_axis, smooth_validation_y_axis = self.smooth_data(self.validate_loss, epochs)

        plt.semilogy(smooth_train_x_axis, smooth_train_y_axis, color="blue", linewidth=1, label="train")
        plt.semilogy(smooth_validation_x_axis, smooth_validation_y_axis, color="red", linewidth=1, label="validation")

        plt.title("model loss")
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.legend()

        plt.ioff()
        plt.show()

    def plot_accuracy(self, epochs):
        smooth_train_x_axis, smooth_train_y_axis = self.smooth_data(self.train_accuracy, epochs)
        smooth_validation_x_axis, smooth_validation_y_axis = self.smooth_data(self.validate_accuracy, epochs,)

        plt.plot(smooth_train_x_axis, smooth_train_y_axis, color="blue", linewidth=1, label="train")
        plt.plot(smooth_validation_x_axis, smooth_validation_y_axis, color="red", linewidth=1, label="validation")

        plt.title("model accuracy")
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.legend()

        plt.ioff()
        plt.show()

    def smooth_data(self, data, epochs):
        window = len(data) // 30
        if window % 2 == 0:
            window -= 1

        smooth_y_axis = savgol_filter(data, window, 0)
        smooth_x_axis = np.arange(0, epochs, epochs / len(smooth_y_axis))

        return smooth_x_axis, smooth_y_axis

    # Input x = Matrix, y = Matrix
    def shuffle(self, x, y):
        # shuffles data in unison with helping from indexing
        indexes = np.random.permutation(x.shape[1])
        return x[..., indexes], y[..., indexes]


if __name__ == "__main__":
    train_data, train_labels, test_data, test_labels = load_mnist()
    net = Network()
    net.addInputLayer(28 * 28)
    net.addFullyConnectedLayer(100, activation="relu")
    net.addFullyConnectedLayer(10, activation="sigmoid")
    net.regression(learning_rate=0.1, cost="quadratic")
    net.fit(train_data, train_labels, test_data, test_labels, epochs=20, mini_batch_size=20, plot=False)
    net.evaluate(test_data, test_labels)
    # best accuracy: 0.9822
