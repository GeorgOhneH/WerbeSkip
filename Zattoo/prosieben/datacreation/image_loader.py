from prosieben.datacreation.create_images import plane_background
import numpy as np


def shuffle(x, y):
    # shuffles data in unison with helping from indexing
    indexes = np.random.permutation(x.shape[1])
    return x[:, indexes], y[:, indexes]


def img_to_matrix(imgs):
    num_imgs = [list(x.getdata()) for x in imgs]
    imgs_matrix = np.matrix(num_imgs).T / 255
    return imgs_matrix


def load_imgs():
    train_image_logo = img_to_matrix(plane_background(True))
    train_label_logo = np.matrix([[0, 1] for _ in range(train_image_logo.shape[1])]).T

    train_image_no_logo = img_to_matrix(plane_background(True))
    train_label_no_logo = np.matrix([[1, 0] for _ in range(train_image_no_logo.shape[1])]).T

    train_image = np.c_[train_image_logo, train_image_no_logo]  # appends the matrix
    train_label = np.c_[train_label_logo, train_label_no_logo]

    train_image, train_label = shuffle(train_image, train_label)

    test_size = int(len(train_image) * 0.1)

    return train_image[:, :-test_size], train_label[:, :-test_size], train_image[:, -test_size:], train_label[:, -test_size:]


if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf)
    print(load_imgs()[0][:, 0].reshape((50, 50)))



