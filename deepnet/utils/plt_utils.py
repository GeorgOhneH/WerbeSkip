from scipy.signal import savgol_filter
import cupy as np
import matplotlib.pyplot as plt

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
        smooth_train_y_axis = self.smooth_data(self.network.train_loss)
        smooth_validation_y_axis = self.smooth_data(self.network.validate_loss)

        plt.semilogy(smooth_train_y_axis, color="blue", linewidth=1, label="train")
        plt.semilogy(smooth_validation_y_axis, color="red", linewidth=1, label="validation")

        plt.title("model loss")
        plt.xlabel("training steps")
        plt.ylabel("loss")
        plt.legend()

        plt.ioff()
        plt.show()

    def plot_accuracy(self):
        """
        smooths the data and plots the accuracy of the validation data and
        the training data
        :return: None
        """
        smooth_train_y_axis = self.smooth_data(self.network.train_accuracy)
        smooth_validation_y_axis = self.smooth_data(self.network.validate_accuracy)

        plt.plot(smooth_train_y_axis, color="blue", linewidth=1, label="train")
        plt.plot(smooth_validation_y_axis, color="red", linewidth=1, label="validation")

        plt.title("model accuracy")
        plt.xlabel("training steps")
        plt.ylabel("accuracy")
        plt.legend()

        plt.ioff()
        plt.show()

    @staticmethod
    def smooth_data(data):
        """
        to smooth the data it uses the savgol filter
        this method is nice, because you can still see
        the variance of data after the smoothing

        :param data: array like
        :return smooth_axis: ndarray
            Smoothed array with the same dimensions
        """
        window = len(data) // 30
        if window % 2 == 0:
            window -= 1

        smooth_axis = savgol_filter(data, window, 1)

        return smooth_axis
