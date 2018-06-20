from image_processing.create_images import plane_background, random_background
from image_processing.cropping_images import sample_imgs
import numpywrapper as np


def shuffle(x, y):
    # shuffles data in unison with helping from indexing
    indexes = np.random.permutation(x.shape[1])
    return x[:, indexes], y[:, indexes]


def img_to_array(imgs):
    imgs = [np.matrix.flatten(x) for x in imgs]
    imgs = np.array(imgs).T
    return imgs


def loader(func):
    image_logo = img_to_array(func(use_logo=True))
    label_logo = np.array([[0, 1] for _ in range(image_logo.shape[1])]).T

    image_no_logo = img_to_array(func(use_logo=False))
    label_no_logo = np.array([[1, 0] for _ in range(image_no_logo.shape[1])]).T

    train_image = np.c_[image_logo, image_no_logo]  # appends the matrix
    train_label = np.c_[label_logo, label_no_logo]

    return shuffle(train_image, train_label)


def load_generator(generator, mini_batch_size):
    dict_labels = {0: [[1], [0]], 1: [[0], [1]]}
    images, labels = None, None
    for _ in generator:
        logo = np.random.randint(0, 2)
        image = generator.send(logo)
        if images is None:
            images = np.reshape(image, (-1, 1))
            labels = np.array(dict_labels[logo])
        else:
            images = np.concatenate((images, np.reshape(image, (-1, 1))), axis=1)
            labels = np.concatenate((labels, dict_labels[logo]), axis=1)
        if images.shape[1] >= mini_batch_size:
            yield (images, labels)
            images, labels = None, None


def load_imgs(split=0.8, mini_batch_size=128, padding=10):
    # generator = load_generator(random_background(padding=padding), mini_batch_size)
    sample_images, sample_labels = loader(sample_imgs)
    split_data = int(sample_images.shape[1] * split)

    return 1, \
           sample_images[..., :split_data], sample_labels[..., :split_data], \
           sample_images[..., -split_data:], sample_labels[..., -split_data:]


if __name__ == "__main__":
    gen = load_generator(random_background(), 100)
    for imgs, labels in gen:
        print(labels.shape)
