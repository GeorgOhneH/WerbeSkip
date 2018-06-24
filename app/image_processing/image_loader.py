import os
import cv2
import numpy as np
from tqdm import tqdm
from deepnet.utils import shuffle

# Rations:
# No Boarder: 16:9
# Boarder left right: 8.5:11
# Boarder above below: 2.25:1

# coordinates of the position of the middle of the logo

DICTIONARIES = [
    {"path": "prosieben/images/classified/logo",
     "cords": (922, 49),
     "name": "logo", },
    {"path": "prosieben/images/classified/logo_boarder_above_below",
     "cords": (922, 87),
     "name": "logo_boarder_above_below", },
    {"path": "prosieben/images/classified/logo_boarder_left_right",
     "cords": (807, 49),
     "name": "logo_boarder_left_right", },
    {"path": "prosieben/images/classified/no_logo",
     "cords": None,
     "name": "no_logo", },
]


def _get_img(path_to_img, cords, padding_w, padding_h):
    padding_w += 16
    padding_h += 16
    img = cv2.imread(path_to_img)
    if not cords:
        cords = (865, 68)
    x_middle, y_middle = cords
    img = img[y_middle - padding_h:y_middle + padding_h, x_middle - padding_w:x_middle + padding_w]

    img = np.transpose(img, (2, 0, 1)).astype(dtype="float32") / 255
    return np.expand_dims(img, axis=0)


def load_ads_cnn(split=0.8, padding_w=10, padding_h=10, center=False):
    inputs = []
    labels = []
    print("Load Images. Total Directories: {}".format(len(DICTIONARIES)))
    for dictionary in DICTIONARIES:
        path = dictionary["path"]
        path = os.path.join(os.path.split(os.path.dirname(__file__))[0], path)
        cords = dictionary["cords"]
        if center:
            cords = None
        for img_name in tqdm(os.listdir(path), desc=dictionary["name"]):
            inputs.append(_get_img(os.path.join(path, img_name), cords, padding_w, padding_h))
            if dictionary["cords"]:
                labels.append([[0, 1]])
            else:
                labels.append([[1, 0]])
    inputs = np.concatenate(inputs, axis=0)
    labels = np.concatenate(labels, axis=0)
    inputs, labels = shuffle(inputs, labels)

    split = int(inputs.shape[0] * split)

    return inputs[:split], labels[:split], inputs[split:], labels[split:]


if __name__ == "__main__":
    v_x, v_y, t_x, t_y = load_ads_cnn(split=0.2)
    print(v_x.shape, v_y.shape, t_x.shape, t_y.shape)
