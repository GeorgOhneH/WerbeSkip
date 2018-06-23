import cv2
import zipfile
import random
import numpy as np
import requests
from requests.exceptions import ConnectTimeout, ConnectionError, HTTPError
import warnings
import os
from deepnet.utils import Generator


class TrainGenerator(Generator):
    def __init__(self, epochs, mini_batch_size, padding_w, padding_h, n_workers=1):
        self.logo = None
        self.part_w = None
        self.part_h = None
        self.PATH_TO_LOGO = os.path.join(os.path.split(os.path.dirname(__file__))[0],
                                         "prosieben/images/important_images/logo32x32.png")
        self.PATH_TO_URLS = os.path.join(os.path.dirname(__file__), "urls.zip")
        self.urls = []
        self.dict_labels = {0: [[1], [0]], 1: [[0], [1]]}
        self.padding_h = padding_w
        self.padding_w = padding_h
        self.init()
        super().__init__(epochs, mini_batch_size, n_workers)

    def init(self):
        logo = cv2.imread(self.PATH_TO_LOGO)
        # normalize image
        self.logo = logo.astype("float32") / 255

        logo_w, logo_h, logo_depth = logo.shape
        self.part_w, self.part_h = logo_w + self.padding_w * 2, logo_h + self.padding_h * 2  # size of returning images

        with zipfile.ZipFile(self.PATH_TO_URLS, "r") as archive:
            data = archive.read("urls.txt")
        self.urls = data.decode('UTF-8').split("\n")

        random.shuffle(self.urls)

    def __len__(self):
        return len(self.urls)

    def cubify(self, arr, newshape):
        """https://stackoverflow.com/questions/42297115/numpy-split-cube-into-cubes/42298440#42298440"""
        oldshape = np.array(arr.shape)
        repeats = (oldshape / newshape).astype(int)
        tmpshape = np.column_stack([repeats, newshape]).ravel()
        order = np.arange(len(tmpshape))
        order = np.concatenate([order[::2], order[1::2]])
        # newshape must divide oldshape evenly or else ValueError will be raised
        return arr.reshape(tmpshape).transpose(order).reshape(-1, *newshape)

    def get_mini_batches(self, index):
        url = self.urls[index]
        mini_batches = []
        url = url.strip()

        # load img from url
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            image = np.asarray(bytearray(response.content), dtype="uint8")
            image = cv2.imdecode(image, 1)

            # normalize image
            image = image.astype("float32") / 255

            # cuts the end of the image so it is even dividable
            w, h, d = image.shape
            image = image[:w - w % self.part_w, :h - h % self.part_h]

            # cuts the image in smaller pieces
            image_parts = self.cubify(image, (self.part_w, self.part_h, d))

            for image_part in image_parts:
                use_logo = np.random.randint(0, 2)
                if use_logo:
                    # sets logo in a random place of the image
                    pad_w = np.random.randint(0, self.padding_w * 2)
                    pad_h = np.random.randint(0, self.padding_h * 2)
                    logo_padding = np.pad(self.logo, [(pad_w, self.padding_w * 2 - pad_w),
                                                      (pad_h, self.padding_h * 2 - pad_h),
                                                      (0, 0)],
                                          "constant")
                    # applies screen blend effect
                    image_part = 1 - (1 - logo_padding) * (1 - image_part)

                images = np.expand_dims(np.transpose(image_part, (2, 0, 1)), axis=0)
                labels = np.array(self.dict_labels[use_logo]).reshape((1, -1))
                mini_batches.append((images, labels))

        except ConnectTimeout or ConnectionError or HTTPError as e:
            warnings.warn("A request wasn't successful, got {}".format(e))
        return mini_batches
