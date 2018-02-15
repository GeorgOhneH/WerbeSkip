from scipy.signal import savgol_filter
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')  # for nicer looking plotting


class Plotter(object):
    """The Plotter class is responsible to plot """
    def __init__(self, train_loss, validate_loss, train_accuracy, validate_accuracy):
        """init the needed references"""
        self.train_loss = train_loss
        self.validate_loss = validate_loss
        self.train_accuracy = train_accuracy
        self.validate_accuracy = validate_accuracy

    def plot_loss(self, epochs):
        """
        smooths the data and plots the loss of the validation data and
        the training data
        It uses a semilogy scale
        :param epochs: int
        :return: None
        """
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
        """
        smooths the data and plots the accuracy of the validation data and
        the training data
        :param epochs: unsigned int
        :return: None
        """
        smooth_train_x_axis, smooth_train_y_axis = self.smooth_data(self.train_accuracy, epochs)
        smooth_validation_x_axis, smooth_validation_y_axis = self.smooth_data(self.validate_accuracy, epochs, )

        plt.plot(smooth_train_x_axis, smooth_train_y_axis, color="blue", linewidth=1, label="train")
        plt.plot(smooth_validation_x_axis, smooth_validation_y_axis, color="red", linewidth=1, label="validation")

        plt.title("model accuracy")
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.legend()

        plt.ioff()
        plt.show()

    @staticmethod
    def smooth_data(data, epochs):
        """
        to smooth the data it uses the savgol filter
        this method is nice, because you can still see
        the variance of data after the smoothing

        :param data: array like
        :param epochs: unsigned int
        :return smooth_x_axis: ndarray
        :return smooth_y_axis: ndarray
        """
        window = len(data) // 30
        if window % 2 == 0:
            window -= 1

        smooth_y_axis = savgol_filter(data, window, 0)
        smooth_x_axis = np.arange(0, epochs, epochs / len(smooth_y_axis))

        return smooth_x_axis, smooth_y_axis
