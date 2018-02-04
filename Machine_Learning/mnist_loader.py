import numpy as np

# mnist loader from https://github.com/sorki/python-mnist
from mnist import MNIST


def load_mnist():
    train_images, train_labels = MNIST.load("mnist/train-images.idx3-ubyte", "mnist/train-labels.idx1-ubyte")
    test_images, test_labels = MNIST.load("mnist/t10k-images.idx3-ubyte", "mnist/t10k-labels.idx1-ubyte")

    np_train_images = np.matrix(train_images, dtype="float16").transpose() / 255
    np_train_labels = np.asmatrix(np.zeros((10, len(train_labels)), dtype="int8"))
    for index, num in enumerate(train_labels):
        np_train_labels[num, index] = 1

    np_test_images = np.matrix(test_images, dtype="float16").transpose() / 255
    np_test_labels = np.asmatrix(np.zeros((10, len(test_labels)), dtype="int8"))
    for index, num in enumerate(test_labels):
        np_test_labels[num, index] = 1
    return np_train_images, np_train_labels, np_test_images, np_test_labels


if __name__ == "__main__":
    train_data, train_labels, test_data, test_labels = load_mnist()
    print(train_data.shape, train_labels.shape, test_data.shape, test_labels.shape)
