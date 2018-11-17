import cv2
import zipfile
import random
import numpy as np
import requests
import warnings
import os
from deepnet.utils import Generator
from deepnet import Network
import cv2
from helperfunctions.image_processing.image_loader import load_ads_cnn
from helperfunctions.image_processing.retrieving_images import VideoCapture
from deepnet.layers import FullyConnectedLayer, BatchNorm, Dropout, ReLU, SoftMax, ConvolutionLayer, MaxPoolLayer, \
    Flatten
from deepnet.optimizers import Adam, SGD
from deepnet.functions.costs import QuadraticCost, CrossEntropyCost
import time
import cupy
import deepdish as dd


class LogoGenerator(Generator):
    """
    makes images with and without logo and labels them
    """
    def __init__(self, epochs, mini_batch_size, padding_w, padding_h, n_workers=1,
                 channel="zattoo", colour=True, buffer_multiplier=2, test_white_square=False):
        CHANNELS = {
            "zattoo": "prosieben/images/zattoo/important_images/logo32x32.png",
            "teleboy": "prosieben/images/teleboy/important_images/logo17x11.png",
        }
        self.test_white_square = test_white_square
        self.logo = None
        self.part_w = None
        self.part_h = None
        self.PATH_TO_LOGO = os.path.join(os.path.split(os.path.dirname(__file__))[0], CHANNELS[channel])
        self.PATH_TO_URLS = os.path.join(os.path.dirname(__file__), "urls.zip")
        self.urls = []
        self.dict_labels = {0: [[1], [0]], 1: [[0], [1]]}
        self.padding_w = int(padding_w * 2)
        self.padding_h = int(padding_h * 2)
        self.colour = colour
        self.init()
        super().__init__(epochs, mini_batch_size, n_workers, buffer_multiplier)

    def init(self):
        if self.colour:
            logo = cv2.imread(self.PATH_TO_LOGO)
        else:
            logo = np.expand_dims(cv2.imread(self.PATH_TO_LOGO, 0), axis=2)

        if self.test_white_square:
            logo = np.zeros_like(logo)
            logo[:, :11, :] += 255

        # normalize image
        self.logo = logo.astype("float32") / 255

        logo_h, logo_w, logo_depth = logo.shape
        self.part_h, self.part_w = logo_h + self.padding_h, logo_w + self.padding_w  # size of returning images

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
        out = arr.reshape(tmpshape).transpose(order).reshape(-1, *newshape)
        return out

    def get_mini_batches(self, index):
        url = self.urls[index]
        mini_batches = []
        url = url.strip()

        # load img from url
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            image = np.asarray(bytearray(response.content), dtype="uint8")
            if self.colour:
                image = cv2.imdecode(image, 1)
            else:
                image = np.expand_dims(cv2.imdecode(image, 0), axis=2)

            # normalize image
            image = image.astype("float32") / 255

            # cuts the end of the image so it is even dividable
            h, w, d = image.shape
            image = image[:h - h % self.part_h, :w - w % self.part_w]

            # cuts the image in smaller pieces
            image_parts = self.cubify(image, (self.part_h, self.part_w, d))

            for image_part in image_parts:
                use_logo = np.random.randint(0, 2)
                if use_logo:
                    for _ in range(10):
                        # sets logo in a random place of the image
                        pad_h = np.random.randint(0, self.padding_h)
                        pad_w = np.random.randint(0, self.padding_w)
                        logo_padding = np.pad(self.logo, [(pad_h, self.padding_h - pad_h),
                                                          (pad_w, self.padding_w - pad_w),
                                                          (0, 0)],
                                              "constant")
                        # applies screen blend effect
                        image_part = 1 - (1 - logo_padding) * (1 - image_part)
                images = np.expand_dims(np.transpose(image_part, (2, 0, 1)), axis=0)
                labels = np.array(self.dict_labels[use_logo]).reshape((1, -1))
                mini_batches.append((images, labels))

        except Exception as e:
            warnings.warn("A request wasn't successful, got {}".format(e))
        return mini_batches


if __name__ == "__main__":
    import time

    net = Network()

    net.use_gpu = True

    net.input((1, 180, 320))

    net.add(ConvolutionLayer(n_filter=16, width_filter=12, height_filter=8, stride=4))
    net.add(BatchNorm())
    net.add(ReLU())
    net.add(ConvolutionLayer(n_filter=64, width_filter=6, height_filter=4, stride=1))
    net.add(BatchNorm())
    net.add(ReLU())
    net.add(MaxPoolLayer(width_filter=3, height_filter=3, stride=2))
    net.add(ConvolutionLayer(n_filter=128, width_filter=4, height_filter=4, stride=1))
    net.add(BatchNorm())
    net.add(ReLU())
    net.add(ConvolutionLayer(n_filter=256, width_filter=3, height_filter=3, stride=2))
    net.add(BatchNorm())
    net.add(ReLU())
    net.add(ConvolutionLayer(n_filter=512, width_filter=3, height_filter=3, stride=1))
    net.add(BatchNorm())
    net.add(Dropout(0.8))
    net.add(Flatten())
    net.add(FullyConnectedLayer(2048))
    net.add(BatchNorm())
    net.add(ReLU())
    net.add(Dropout(0.8))
    net.add(FullyConnectedLayer(2))
    net.add(SoftMax())

    optimizer = Adam(learning_rate=0.01)
    net.regression(optimizer=optimizer, cost=CrossEntropyCost())
    net.load("C:\Jetbrains\PyCharm\WerbeSkip\helperfunctions\prosieben\\networks\\teleboy\\teleboy.h5")
    generator = LogoGenerator(epochs=1, mini_batch_size=100, padding_w=151.5, padding_h=84.5, colour=False, channel="teleboy", test_white_square=False)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    losses = []
    for mini_batch in generator:
        x, y = cupy.asnumpy(net.feedforward(mini_batch[0])), cupy.asnumpy(mini_batch[1])
        losses.append(float(net._cost.function(cupy.asarray(x), cupy.asarray(y))))
        a = x.argmax(axis=1) + y.argmax(axis=1) * 2
        tp += int(np.count_nonzero(a == 3))  # True Positive
        tn += int(np.count_nonzero(a == 0))  # True Negative
        fp += int(np.count_nonzero(a == 1))  # False Positive
        fn += int(np.count_nonzero(a == 2))  # False Negative
        print(tp, tn, fp, fn, np.mean(losses))
