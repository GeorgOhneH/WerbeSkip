from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from deepnet.utils import make_batches
import numpy as np

plt.style.use('ggplot')  # for nicer looking plotting


class Plotter(object):
    """The Plotter class is responsible to plot """
    def __init__(self, network):
        """links the network"""
        self.network = network

    def plot_loss(self):
        """
        smooths the data and plots the loss of the validation data and
        the training data
        It uses a semilogy scale
        :return: None
        """
        smooth_train_y_axis, train_x_axis = self.smooth_data(self.network.train_loss)
        smooth_validation_y_axis, validation_x_axis = self.smooth_data(self.network.validate_loss)

        plt.plot(train_x_axis, smooth_train_y_axis, color="blue", linewidth=1, label="train loss")
        if smooth_validation_y_axis:
            plt.plot(validation_x_axis, smooth_validation_y_axis, color="red", linewidth=1, label="validation")

        plt.ylabel("loss")
        # plt.title("model loss")
        plt.xlabel("Trainingsschritte")
        plt.legend()

        plt.ioff()
        plt.show()

    def plot_accuracy(self):
        """
        smooths the data and plots the accuracy of the validation data and
        the training data
        :return: None
        """
        smooth_train_y_axis, train_x_axis = self.smooth_data(self.network.train_accuracy)
        smooth_validation_y_axis, validation_x_axis = self.smooth_data(self.network.validate_accuracy)

        plt.plot(train_x_axis, smooth_train_y_axis, color="blue", linewidth=1, label="train mcc")
        if smooth_validation_y_axis:
            plt.plot(validation_x_axis, smooth_validation_y_axis, color="red", linewidth=1, label="validation")

        plt.ylabel("MCC")
        # plt.title("model accuracy")
        plt.xlabel("Trainingsschritte")
        plt.legend()

        plt.ioff()
        plt.show()

    @staticmethod
    def smooth_data(data):
        """
        means the data
        :param data: array like
        :return smooth_axis: ndarray
            Smoothed array with the same dimensions
        """
        if not data:
            return data, np.array([])

        batch_size = len(data) // 100

        smooth_axis = [np.mean(batch) for batch in make_batches(data, batch_size)]
        x_axis = np.arange(len(smooth_axis)) * len(data) / len(smooth_axis)

        return smooth_axis, x_axis
