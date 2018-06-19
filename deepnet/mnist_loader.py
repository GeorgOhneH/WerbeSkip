import cupy as np

# mnist loader from https://github.com/sorki/python-mnist
from mnist import MNIST


def load_mnist():
    mn = MNIST("mnist")
    train_images, train_labels = mn.load_training()
    test_images, test_labels = mn.load_testing()

    np_train_images = np.array(train_images, dtype="float32").transpose() / 255
    np_train_labels = np.asarray(np.zeros((10, len(train_labels)), dtype="int8"))
    for index, num in enumerate(train_labels):
        np_train_labels[num, index] = 1

    np_test_images = np.array(test_images, dtype="float32").transpose() / 255
    np_test_labels = np.asarray(np.zeros((10, len(test_labels)), dtype="int8"))
    for index, num in enumerate(test_labels):
        np_test_labels[num, index] = 1
    return np_train_images, np_train_labels, np_test_images, np_test_labels


def load_conv():
    X, Y, test_x, test_y = load_mnist()
    X = X.T
    Y = Y.T
    test_x = test_x.T
    test_y = test_y.T
    X = X.reshape([-1, 1, 28, 28])
    test_x = test_x.reshape([-1, 1, 28, 28])
    return X, Y, test_x, test_y


if __name__ == "__main__":
    train_data, train_labels, test_data, test_labels = load_conv()
    print(train_data.shape, train_labels.shape, test_data.shape, test_labels.shape)
