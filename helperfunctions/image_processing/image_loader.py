import os
import cv2
import numpy as np
from tqdm import tqdm
import deepdish as dd
from deepnet.utils import shuffle
import random

# Rations:
# No Boarder: 16:9
# Boarder left right: 8.5:11
# Boarder above below: 2.25:1

# coordinates of the position of the middle of the logo

DICTIONARIES = [
    {"path": "prosieben/images/zattoo/classified/logo",
     "cords": (922, 49),
     "name": "logo", },
    {"path": "prosieben/images/zattoo/classified/logo_boarder_above_below",
     "cords": (922, 87),
     "name": "logo_boarder_above_below", },
    {"path": "prosieben/images/zattoo/classified/logo_boarder_left_right",
     "cords": (807, 49),
     "name": "logo_boarder_left_right", },
    {"path": "prosieben/images/zattoo/classified/no_logo",
     "cords": None,
     "name": "no_logo", },
    # {"path": "prosieben/images/zattoo/classified/special",
    #  "cords": (0, 0),
    #  "name": "special", },
]


def _get_img(path_to_img, cords, padding_w, padding_h, colour, full, cropped):
    padding_w += 16
    padding_h += 16
    if colour:
        img = cv2.imread(path_to_img)
    else:
        img = cv2.imread(path_to_img, 0)

    if not full:
        if not colour:
            img = np.expand_dims(img, axis=2)
        if not cords:
            cords = (865, 68)  # middle of all 3 possible positions
        x_middle, y_middle = cords

        img = np.pad(img, [(padding_h, padding_h), (padding_w, padding_w), (0, 0)], mode="constant", constant_values=255)
        img = img[y_middle:y_middle + 2*padding_h, x_middle:x_middle + 2*padding_w]
    else:
        img = cv2.resize(img, (320, 180), cv2.INTER_AREA)
        if not colour:
            img = np.expand_dims(img, axis=2)

    if cropped and full:
        img = img[38:-1, 0:269, :]

    img = np.transpose(img, (2, 0, 1)).astype(dtype="float32") / 255
    return np.expand_dims(img, axis=0)


def _get_path_to_file(padding_w, padding_h, center, colour, full, volume, cropped):
    path_to_cache = os.path.join(os.path.dirname(__file__), "load_cache")
    file_name = "w{}_h{}_ce{}_co{}_f{}_v{}_cp{}.h5".format(padding_w, padding_h, center, colour, full, volume, cropped)
    path_to_file = os.path.join(path_to_cache, file_name)
    return path_to_file


def _load_cache(path_to_file):
    if os.path.exists(path_to_file):
        inputs, labels = dd.io.load(path_to_file)
        return inputs, labels
    return np.array([]), np.array([])


def _save_cache(path_to_file, inputs, labels):
    if not os.path.exists(path_to_file):
        dd.io.save(path_to_file, [inputs, labels])
        return True
    return False


def _load_imgs(padding_w, padding_h, center, colour, full, volume, cropped):
    inputs = []
    labels = []
    print("Load Images. Total Directories: {}".format(len(DICTIONARIES)))
    for dictionary in DICTIONARIES:
        path = dictionary["path"]
        path = os.path.join(os.path.split(os.path.dirname(__file__))[0], path)
        cords = dictionary["cords"]
        if center:
            cords = None
        files = os.listdir(path)
        random.shuffle(files)
        files = files[:int(len(files)*volume)]
        for img_name in tqdm(files, desc=dictionary["name"]):
            inputs.append(_get_img(os.path.join(path, img_name), cords, padding_w, padding_h, colour, full, cropped))
            if dictionary["cords"]:
                labels.append([[0, 1]])
            else:
                labels.append([[1, 0]])
    inputs = np.concatenate(inputs, axis=0)
    labels = np.concatenate(labels, axis=0)
    return inputs, labels


def load_ads_cnn(split=0.8, padding_w=10, padding_h=10, center=False,
                 cache=False, colour=True, full=False, shuffle_set=True, volume=1., cropped=False):
    """
    loads prosieben images
    :param split: spits image to validate and test set
    :param padding_w: width
    :param padding_h: height
    :param center: use middle of the three logos
    :param cache: save to cache
    :param colour: colour or not
    :param full: full image padding and center doesn't have an effect
    :param shuffle_set: shuffles set
    :param volume: percentage used of directories
    :param cropped: for ads generator
    :return: (v_x, v_y, t_x, t_y)
    """
    path_to_file = _get_path_to_file(padding_w, padding_h, center, colour, full, volume, cropped)

    inputs, labels = _load_cache(path_to_file)
    if inputs.shape[0] == 0 and labels.shape[0] == 0:
        inputs, labels = _load_imgs(padding_w, padding_h, center, colour, full, volume, cropped)
        if cache:
            _save_cache(path_to_file, inputs, labels)

    if shuffle_set:
        inputs, labels = shuffle(inputs, labels)

    split = int(inputs.shape[0] * split)

    return inputs[:split], labels[:split], inputs[split:], labels[split:]


if __name__ == "__main__":
    v_x, v_y, t_x, t_y = load_ads_cnn(split=0.2, full=True, cropped=True, volume=0.1)
    print(v_x.shape, v_y.shape, t_x.shape, t_y.shape)
