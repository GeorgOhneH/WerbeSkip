import cv2
import zipfile
import random
import numpy as np
import requests
from requests.exceptions import ConnectTimeout, ConnectionError, HTTPError
import warnings
from deepnet.utils import Generator, blockshaped


class TrainGenerator(Generator):
    def __init__(self, epochs, mini_batch_size, padding, n_workers):
        self.logo = None
        self.part_w = None
        self.part_h = None
        self.urls = []
        self.dict_labels = {0: [[1], [0]], 1: [[0], [1]]}
        self.padding = padding
        self.init()
        super().__init__(epochs, mini_batch_size, n_workers)

    def init(self):
        logo = cv2.imread("../prosieben/images/important_images/logo32x32.png", 0)  # 0 is the mode for white/black
        # normalize image
        self.logo = logo.astype(float) / 255

        logo_w, logo_h = logo.shape
        self.part_w, self.part_h = logo_w + self.padding * 2, logo_h + self.padding * 2  # size of returning images

        with zipfile.ZipFile("../image_processing/urls.zip", "r") as archive:
            data = archive.read("urls.txt")
        self.urls = data.decode('UTF-8').split("\n")

        random.shuffle(self.urls)

    def __len__(self):
        return len(self.urls)

    def get_mini_batches(self, index):
        url = self.urls[index]
        mini_batches = []
        url = url.strip()

        # load img from url
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            image = np.asarray(bytearray(response.content), dtype="uint8")
            image = cv2.imdecode(image, 0)

            # normalize image
            image = image.astype(float) / 255

            # cuts the end of the image so it is even dividable
            w, h = image.shape
            image = image[:w - w % self.part_w, :h - h % self.part_h]

            # cuts the image in smaller pieces
            image_parts = blockshaped(image, self.part_w, self.part_h)

            for image_part in image_parts:
                use_logo = np.random.randint(0, 2)
                if use_logo:
                    # sets logo in a random place of the image
                    pad_w = np.random.randint(0, self.padding * 2)
                    pad_h = np.random.randint(0, self.padding * 2)
                    logo_padding = np.pad(self.logo, [(pad_w, self.padding * 2 - pad_w),
                                                      (pad_h, self.padding * 2 - pad_h)],
                                          "constant")
                    # applies screen blend effect
                    image_part = 1 - (1 - logo_padding) * (1 - image_part)

                images = np.reshape(image_part, (-1, 1))
                labels = np.array(self.dict_labels[use_logo])
                mini_batches.append((images, labels))

        except ConnectTimeout or ConnectionError or HTTPError as e:
            warnings.warn("A request wasn't successful, got {}".format(e))
        return mini_batches

