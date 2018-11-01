import numpywrapper as np
import math


class Analysis(object):
    """
    The class is responsible for the evaluation
    and the validation of the network
    """

    def __init__(self, network):
        """
        The class is linked with the network
        :param network: network object
        """
        self.network = network

    @staticmethod
    def accuracy(x, y):
        """
        Computes the accuracy with values, that went already
        through the network
        :param x: ndarray
        :param y: ndarray
        :return: accuracy: flout
        """
        return np.mean(np.argmax(x, axis=1) == np.argmax(y, axis=1))

    @staticmethod
    def get_mcc(x, y):
        """
        after https://en.wikipedia.org/wiki/Matthews_correlation_coefficient
        Computes the accuracy with values, that went already
        through the network
        :param x: ndarray
        :param y: ndarray
        :return: accuracy: flout
        """
        a = np.argmax(x, axis=1) + np.argmax(y, axis=1) * 2
        tp = int(np.count_nonzero(a == 3))  # True Positive
        tn = int(np.count_nonzero(a == 0))  # True Negative
        fp = int(np.count_nonzero(a == 1))  # False Positive
        fn = int(np.count_nonzero(a == 2))  # False Negative
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        mcc = (tp * tn - fp * fn) / (math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + 1e-8)
        return mcc, accuracy

    def accuracy_or_mcc(self, x, y):
        if self.network.is_binary:
            return self.get_mcc(x, y)[0]
        else:
            return self.accuracy(x, y)

    @staticmethod
    def wrong_predictions(x, y):
        """
        Calculates all inputs that were not correctly predicted and
        returns them as indexes in a ndarray

        :param x: ndarray
        :param y: ndarray
        :return: indexes: ndarray
        """
        indexes = np.nonzero(np.argmax(x) == np.argmax(y) - 1)
        return indexes

    def validate(self, x, y, size=None):
        """
        calculates loss and accuracy of the validation set
        and saves them in arrays

        Normally the hole validation set  runs through the
        network unless you set the size, then will run a random
        portion of the size through the network

        size is normally the same as the mini batch while training

        :param x: ndarray
        :param y: ndarray
        :param size: unsigned int
        :return loss: flout
        :return accuracy: flout
        """
        x = np.asarray(x)
        y = np.asarray(y)

        if size is not None:
            rand = np.random.randint(0, x.shape[0] - size)
            x = x[rand:rand + size]
            y = y[rand:rand + size]

        x = self.network.feedforward(x)

        loss = self.network.cost.function(x, y)
        accuracy = self.accuracy_or_mcc(x, y)

        self.network.validate_accuracy.append(float(accuracy))
        self.network.validate_loss.append(float(loss))
        return loss, accuracy

    def evaluate(self, x, y):
        """
        Evaluates the Network with the test set

        It has 2 modes:
        -The normal evaluation:
            It's the default mode
        -The binary evaluation:
            This mode will be called if the output
            of the network has the size of 2
            this means it can be answered with
            Yes no No

        :param x: ndarray
        :param y: ndarray
        :return: None
        """
        x = np.asarray(x)
        y = np.asarray(y)

        x = self.network.feedforward(x)
        loss = self.network.cost.function(x, y)
        if not self.network.is_binary:
            self.normal_evaluation(x, y, loss)
        else:
            self.binary_evaluation(x, y, loss)

    def normal_evaluation(self, x, y, loss):
        """
        computes the accuracy and then prints the
        loss and the accuracy

        :param x: ndarray
        :param y: ndarray
        :param loss: flout
        :return: print: results
        """
        accuracy = self.accuracy(x, y)
        print("Evaluation of {} inputs:\n"
              "loss: {:.5f} | accuracy: {:.5f}".format(
            int(x.shape[0]), float(loss), float(accuracy),
        ))

    def binary_evaluation(self, x, y, loss):
        """
        computes the accuracy. the precision, the recall
        and the mcc and prints them out

        :param x: ndarray
        :param y: ndarray
        :param loss: flout
        :return: print: results
        """
        mcc, accuracy = self.get_mcc(x, y)
        print("Evaluation of {} inputs:\n"
              "loss: {:.5f} | accuracy: {:.5f} | MCC: {:.5f}".format(
            int(x.shape[0]), float(loss), float(accuracy), float(mcc)
        ))
