from deepnet import Network
import cv2
import numpy as np
from helperfunctions.image_processing.image_loader import load_ads_cnn
from helperfunctions.image_processing.logo_generator import LogoGenerator
from helperfunctions.image_processing.retrieving_images import VideoCapture
from deepnet.layers import FullyConnectedLayer, BatchNorm, Dropout, ReLU, SoftMax, ConvolutionLayer, MaxPoolLayer, \
    Flatten
from deepnet.optimizers import Adam, SGD
from deepnet.functions.costs import QuadraticCost, CrossEntropyCost
import time
import deepdish as dd

if __name__ == "__main__":
    # np.seterr(invalid='raise')

    gen = LogoGenerator(epochs=1, mini_batch_size=80, padding_w=151.5, padding_h=84.5, colour=False, channel="teleboy", buffer_multiplier=10)

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
    net.load("teleboy.h5")
    net.print_infos()

    # Number of inputs: 2540000, Training time: 69479

    # v_x, v_y, t_x, t_y = load_ads_cnn(split=0, full=True, shuffle_set=False, colour=False)
    # net.evaluate(t_x, t_y)
    # Evaluation of 7830 inputs:
    # loss: 0.22138 | accuracy: 0.91609 | MCC: 0.80169

    result = []
    x = 0
    for img in VideoCapture(channel=354, colour=False, rate_limit=1):
        # cv2.imshow("image", img)
        img = np.expand_dims(img.transpose(2, 0, 1), axis=0)
        value = net.feedforward(img)[0, 1]
        result.append(value)
        if x % 1000 == 0:
            dd.io.save("mean_points2.h5", result)
        print("{}".format(value))
        # cv2.waitKey(1)
        x += 1
