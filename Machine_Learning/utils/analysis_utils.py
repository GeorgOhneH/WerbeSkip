import numpy as np


class Analysis(object):
    def __init__(self, network):
        self.network = network

    def validate(self, x, y, size=None):
        if size is not None:
            rand = np.random.randint(0, x.shape[1] - size)
            x = x[..., rand:rand + size]
            y = y[..., rand:rand + size]

        x = self.network.feedforward(x)

        loss = self.network.cost.function(x, y)
        accuracy = self.accuracy(x, y)

        self.network.validate_accuracy.append(accuracy)
        self.network.validate_loss.append(loss)
        return loss, accuracy

    @staticmethod
    def accuracy(x, y):
        return np.mean(np.argmax(x, axis=0) == np.argmax(y, axis=0))

    def evaluate(self, x, y):
        x = self.network.feedforward(x)
        loss = self.network.cost.function(x, y)
        if x.shape[0] != 2:
            accuracy = self.accuracy(x, y)
            print("Evaluation with {} data:\n"
                  "loss: {:.5f} | accuracy: {:.5f}".format(
                x.shape[1], loss, accuracy,
            ))
            return

        a = np.argmax(x, axis=0) + np.argmax(y, axis=0) * 2
        tp = np.count_nonzero(a == 3)  # True Positive
        tn = np.count_nonzero(a == 0)  # True Negative
        fp = np.count_nonzero(a == 1)  # False Positive
        fn = np.count_nonzero(a == 2)  # False Negative
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * (recall * precision) / (recall + precision)
        print("Evaluation with {} data:\n"
              "loss: {:.5f} | accuracy: {:.5f} | precision: {:.5f} | recall: {:.5f} | f1_score: {:.5f}".format(
            x.shape[1], loss, accuracy, precision, recall, f1_score
        ))
