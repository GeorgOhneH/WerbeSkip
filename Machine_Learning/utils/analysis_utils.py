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
        x = self.network.feedforward(x)
        loss = self.network.cost.function(x, y)
        if x.shape[0] != 2:
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