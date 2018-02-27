from PIL import Image, ImageChops, ImageOps
import cv2
import requests
from requests.exceptions import ConnectTimeout, ConnectionError, HTTPError
import numpy as np
import zipfile
import io
import warnings
import random


def blockshaped(arr, nrows, ncols):
    """
    https://stackoverflow.com/questions/16856788/slice-2d-array-into-smaller-2d-arrays
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    try:
        blocks = (arr.reshape(h // nrows, nrows, -1, ncols)
                  .swapaxes(1, 2)
                  .reshape(-1, nrows, ncols))
    except ValueError:
        blocks = []

    return blocks


def plane_background(padding=10, use_logo=True):
    imgs = []
    borders = [
        (padding, padding, padding, padding),
    ]
    logo = Image.open("../prosieben/images/important_images/logo32x32.png")
    logo = logo.convert(mode="L")  # mode L is white and black
    for border in borders:
        exp_logo = ImageOps.expand(logo, border, fill="black")
        for color in range(0, 224):
            img = Image.new("L", color=color, size=exp_logo.size)
            if use_logo:
                img = ImageChops.screen(exp_logo, img)
            imgs.append(np.array(img) / 255)
    return imgs


def random_background(padding=10, use_logo=True, n_images=40000):
    result = []
    logo = cv2.imread("../prosieben/images/important_images/logo32x32.png", 0)  # 0 is the mode for white/black
    # normalize image
    logo = logo.astype(float) / 255

    logo_w, logo_h = logo.shape
    part_w, part_h = logo_w + padding * 2, logo_h + padding * 2  # size of returning images

    with zipfile.ZipFile("../image_processing/urls.zip", "r") as archive:
        data = archive.read("urls.txt")
    urls = list(io.BytesIO(data))

    random.shuffle(urls)

    for url in urls:
        url = url.strip()

        # load img from url
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            image = np.asarray(bytearray(response.content), dtype="uint8")
            image = cv2.imdecode(image, 0)

            # normalize image
            image = image.astype(float) / 255

            # cuts image so it is even dividable
            w, h = image.shape
            image = image[:w - w % part_w, :h - h % part_h]

            # cuts the image in smaller pieces
            image_parts = blockshaped(image, part_w, part_h)

            for image_part in image_parts:
                if use_logo:
                    # sets logo in a random place of the image
                    pad_w = np.random.randint(0, padding * 2)
                    pad_h = np.random.randint(0, padding * 2)
                    logo_padding = np.pad(logo, [(pad_w, padding * 2 - pad_w), (pad_h, padding * 2 - pad_h)], "constant")
                    # applies screen blend effect
                    image_part = 1. - (1. - logo_padding) * (1. - image_part)

                result.append(image_part)
                if len(result) >= n_images:
                    return result

        except ConnectTimeout or ConnectionError or HTTPError as e:
            warnings.warn("A request wasn't successful, got {}".format(e))

    return result


if __name__ == "__main__":
    imgs = random_background(padding=200, use_logo=True, n_images=40)
    for img in imgs:
        cv2.imshow("test", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
