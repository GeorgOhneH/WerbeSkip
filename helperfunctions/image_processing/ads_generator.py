from deepnet import Network
import numpy as np
import os
from helperfunctions.image_processing.video_capture import VideoCapture
from deepnet.layers import FullyConnectedLayer, BatchNorm, Dropout, ReLU, SoftMax, ConvolutionLayer, MaxPoolLayer, \
    Flatten
from deepnet.optimizers import Adam
from deepnet.functions.costs import CrossEntropyCost

import cv2
from collections import deque
from random import randint


class AdsGenerator(object):
    """
    Generator to crop and label prosieben images
    """
    def __init__(self, epochs, mini_batch_size, ffmpeg_log="error"):
        self.PATH_TO_NET = os.path.join(os.path.split(os.path.dirname(__file__))[0],
                                        "prosieben/networks/teleboy/teleboy.h5")

        self.cap = VideoCapture(channel=354, colour=False, convert_network=True, proxy=False, ffmpeg_log=ffmpeg_log, rate_limit=4)
        self.mini_batch_size = mini_batch_size
        self.epochs = epochs
        self.progress = 0
        self.network = self.init_network()
        self.dict_labels = {0: [1, 0], 1: [0, 1]}

        self.filters = []
        self.result = []
        self.predictions = []
        self.imgs = []

        self.cache_logo = deque([])
        self.cache_logo_future = []
        self.cache_no_logo = deque([])
        self.cache_no_logo_future = []
        self.with_logo = False

        self.filter_size = 25
        self.chain_size = 5
        [self.run() for _ in range(self.filter_size*2)]

    def init_network(self):
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

        optimizer = Adam(learning_rate=0.001)
        net.regression(optimizer=optimizer, cost=CrossEntropyCost())
        net.load(self.PATH_TO_NET)
        return net

    def __iter__(self):
        return self

    def __next__(self):
        inputs = []
        labels = []
        for _ in range(self.mini_batch_size):
            input, label = self.get_mini_batches()
            inputs.append(input)
            labels.append(label)

        return np.array(inputs), np.array(labels)

    def __len__(self):
        return 1000

    def get_mini_batches(self):
        if self.with_logo:
            while len(self.cache_logo) == 0:
                self.use_cache()
                if len(self.cache_logo) == 0 and len(self.cache_logo_future) > 200:
                    result = self.cache_logo_future.pop(randint(0, len(self.cache_logo_future)-1))
                    return result
            result = self.cache_logo.pop()
        else:
            while len(self.cache_no_logo) == 0:
                self.use_cache()
                if len(self.cache_no_logo) == 0 and len(self.cache_no_logo_future) > 200:
                    result = self.cache_no_logo_future.pop(randint(0, len(self.cache_no_logo_future)-1))
                    return result
            result = self.cache_no_logo.pop()

        self.with_logo = not self.with_logo

        return result

    def run(self):
        img = next(self.cap)
        self.imgs.append(img)
        prediction = self.network.feedforward(img)

        self.predictions.append(prediction[0, 1])
        snippet = self.predictions[-self.filter_size:]
        if np.any(np.array(snippet) > 0.9):  # checks if network is sure that it found a logo
            self.filters.append(1)
        else:
            self.filters.append(0)

        last_filter = self.filters[-1]
        if np.all(np.array(self.filters[-self.chain_size:]) == last_filter):  # checks if the last values are the same
            if last_filter == 1:
                if np.mean(self.predictions[-self.chain_size:]) > 0.9:
                    self.result.append(last_filter)
                else:
                    self.result.append(self.result[-1])
            else:
                self.result.append(last_filter)
        else:
            self.result.append(self.result[-1])

        self.clean_up()

    def clean_up(self):
        if len(self.predictions) > 10 * self.filter_size and len(self.filters) > 10 * self.chain_size:
            self.filters.pop(0)
            self.result.pop(0)
            self.predictions.pop(0)
            self.imgs.pop(0)

    def use_cache(self):
        self.run()
        while not np.all(np.array(self.result[-self.filter_size:]) == self.result[-1]):
            self.run()

        result = (self.imgs[-self.filter_size][0, :, 38:-1, 0:269], self.dict_labels[self.result[-self.filter_size]])
        if self.result[-self.filter_size]:
            if len(self.cache_logo) < 4000:
                self.cache_logo.append(result)
                for _ in range(2):
                    if len(self.cache_logo_future) < 4000:
                        self.cache_logo_future.append(result)
        else:
            if len(self.cache_no_logo) < 4000:
                self.cache_no_logo.append(result)
                for _ in range(5):
                    if len(self.cache_no_logo_future) < 4000:
                        self.cache_no_logo_future.append(result)


if __name__ == "__main__":
    for frame in AdsGenerator(1, 10):
        i, l = frame
        for x in range(i.shape[0]):
            cv2.imshow('test', frame[0][x, 0, :, :])
            cv2.waitKey(1)
