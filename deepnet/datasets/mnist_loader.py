import numpy as np
import os

# mnist loader from https://github.com/sorki/python-mnist
from mnist import MNIST


def load_mnist_fc():
    file_name = os.path.join(os.path.dirname(__file__), 'mnist')
    mn = MNIST(file_name)
    train_images, train_labels = mn.load_training()
    test_images, test_labels = mn.load_testing()

    np_train_images = np.array(train_images, dtype="float32") / 255
    np_train_labels = np.asarray(np.zeros((len(train_labels), 10), dtype="int8"))
    for index, num in enumerate(train_labels):
        np_train_labels[index, num] = 1

    np_test_images = np.array(test_images, dtype="float32") / 255
    np_test_labels = np.asarray(np.zeros((len(test_labels), 10), dtype="int8"))
    for index, num in enumerate(test_labels):
        np_test_labels[index, num] = 1
    return np_train_images, np_train_labels, np_test_images, np_test_labels


def load_mnist_cnn():
    x, y, test_x, test_y = load_mnist_fc()
    x = x.reshape([-1, 1, 28, 28])
    test_x = test_x.reshape([-1, 1, 28, 28])
    return x, y, test_x, test_y


if __name__ == "__main__":
    train_data, train_labels, test_data, test_labels = load_mnist_fc()
    print(train_data.shape, train_labels.shape, test_data.shape, test_labels.shape)
