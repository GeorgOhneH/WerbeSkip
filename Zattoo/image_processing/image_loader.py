from image_processing.create_images import plane_background
from image_processing.cropping_images import sample_imgs
import numpy as np


def shuffle(x, y):
    # shuffles data in unison with helping from indexing
    indexes = np.random.permutation(x.shape[1])
    return x[:, indexes], y[:, indexes]


def img_to_matrix(imgs):
    num_imgs = [list(x.getdata()) for x in imgs]
    imgs_matrix = np.matrix(num_imgs).T / 255
    return imgs_matrix


def loader(func):
    image_logo = img_to_matrix(func(True))
    label_logo = np.matrix([[0, 1] for _ in range(image_logo.shape[1])]).T

    image_no_logo = img_to_matrix(func(False))
    label_no_logo = np.matrix([[1, 0] for _ in range(image_no_logo.shape[1])]).T

    train_image = np.c_[image_logo, image_no_logo]  # appends the matrix
    train_label = np.c_[label_logo, label_no_logo]

    return shuffle(train_image, train_label)


def load_imgs(split=0.8):
    train_image, train_label = loader(plane_background)
    sample_images, sample_labels = loader(sample_imgs)
    split_data = int(sample_images.shape[1] * split)

    return train_image, train_label, \
           sample_images[..., :split_data], sample_labels[..., :split_data], \
           sample_images[..., -split_data:], sample_labels[..., -split_data:]


if __name__ == "__main__":
    print(loader(sample_imgs)[0].shape, loader(plane_background)[0].shape)
