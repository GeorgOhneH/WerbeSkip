from PIL import Image
import os
import cupy as np

# Rations:
# No Boarder: 16:9
# Boarder left right: 8.5:11
# Boarder above below: 2.25:1

# coordinates of the position of the middle of the logo
prosieben = {
    "no_boarder": (922, 49),
    "boarder_above_below": (922, 87),
    "boarder_left_right": (807, 49)
}

logo_paths = [
    ("../prosieben/images/classified/logo", "no_boarder"),
    ("../prosieben/images/classified/logo_boarder_above_below", "boarder_above_below"),
    ("../prosieben/images/classified/logo_boarder_left_right", "boarder_left_right"),
]

no_logo_paths = [
    ("../prosieben/images/classified/no_logo", "no_boarder"),
]

PADDING = 26


def get_logo(path_to_img, type):
    img = Image.open(path_to_img).convert(mode="L")
    return np.array(img.crop(
        (prosieben[type][0] - PADDING,
         prosieben[type][1] - PADDING,
         prosieben[type][0] + PADDING,
         prosieben[type][1] + PADDING)
    )) / 255


def sample_imgs(use_logo=True):
    result = []
    if use_logo:
        paths = logo_paths
    else:
        paths = no_logo_paths
    for path in paths:
        for img in os.listdir(path[0]):
            result.append(get_logo(os.path.join(path[0], img), path[1]))
    return result


if __name__ == "__main__":
    print(len(sample_imgs(False)))
