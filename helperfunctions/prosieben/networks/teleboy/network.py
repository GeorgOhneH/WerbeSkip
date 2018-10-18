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
    # training accuracy around 95%
    # training time around 25 hours

    gen = LogoGenerator(epochs=1, mini_batch_size=64, padding_w=151.5, padding_h=84.5, colour=False, channel="teleboy")

    net = Network()

    net.use_gpu = True

    net.input((1, 180, 320))

    net.add(ConvolutionLayer(n_filter=16, width_filter=12, height_filter=8, stride=4, zero_padding=0, padding_value=1))
    net.add(BatchNorm())
    net.add(ReLU())
    net.add(ConvolutionLayer(n_filter=64, width_filter=6, height_filter=4, stride=1, zero_padding=2, padding_value=1))
    net.add(BatchNorm())
    net.add(ReLU())
    net.add(MaxPoolLayer(width_filter=3, height_filter=3, stride=2))
    net.add(ConvolutionLayer(n_filter=128, width_filter=4, height_filter=4, stride=1))
    net.add(BatchNorm())
    net.add(ReLU())
    net.add(ConvolutionLayer(n_filter=128, width_filter=3, height_filter=3, stride=2))
    net.add(BatchNorm())
    net.add(ReLU())
    net.add(ConvolutionLayer(n_filter=256, width_filter=3, height_filter=3, stride=1))
    net.add(BatchNorm())
    net.add(Dropout(0.75))
    net.add(Flatten())
    net.add(FullyConnectedLayer(512))
    net.add(BatchNorm())
    net.add(ReLU())
    net.add(Dropout(0.5))
    net.add(FullyConnectedLayer(2))
    net.add(SoftMax())

    optimizer = Adam(learning_rate=0.01)
    net.regression(optimizer=optimizer, cost=CrossEntropyCost())
    net.load("teleboy.h5")

    v_x, v_y, t_x, t_y = load_ads_cnn(split=0, full=True, shuffle_set=False, colour=False)
    net.evaluate(t_x, t_y)
    # No Boarder
    # Evaluation with 4495 data:
    # loss: 0.16581 | accuracy: 0.90189 | precision: 1.00000 | recall: 0.90189 | f1_score: 0.94842

    # Boarder above, below
    # Evaluation with 813 data:
    # loss: 0.19359 | accuracy: 0.92743 | precision: 1.00000 | recall: 0.92743 | f1_score: 0.96235

    # Boarder side
    # Evaluation with 427 data:
    # loss: 0.10214 | accuracy: 0.94145 | precision: 1.00000 | recall: 0.94145 | f1_score: 0.96984

    # No Logo
    # Evaluation with 2095 data:
    # loss: 0.41319 | accuracy: 0.89833 | precision: 0.00000 | recall: 0.00000 | f1_score: 0.00000

    # All
    # Evaluation with 7830 data:
    # loss: 0.23141 | accuracy: 0.90575 | precision: 0.96072 | recall: 0.90846 | f1_score: 0.93386

    for img in VideoCapture(channel=354, colour=False):
        cv2.imshow("image", img)
        img = np.expand_dims(img.transpose(2, 0, 1), axis=0) / 255
        print("{:.2f}".format(float(net.feedforward(img)[0, 1])))
        cv2.waitKey(1)
