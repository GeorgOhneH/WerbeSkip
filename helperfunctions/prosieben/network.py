from deepnet import Network
import cv2
import numpy as np
from helperfunctions.image_processing.logo_generator import LogoGenerator
from helperfunctions.image_processing.retrieving_images import VideoCapture
from deepnet.layers import FullyConnectedLayer, BatchNorm, Dropout, ReLU, SoftMax, ConvolutionLayer, MaxPoolLayer, Flatten
from deepnet.optimizers import Adam, SGD
from deepnet.functions.costs import CrossEntropyCost
import time
import deepdish as dd

if __name__ == "__main__":
    gen = LogoGenerator(epochs=1, mini_batch_size=64, padding_w=151.5, padding_h=84.5, colour=True, channel="teleboy")

    net = Network()

    net.use_gpu = True

    net.input((3, 180, 320))

    net.add(ConvolutionLayer(n_filter=16, width_filter=8, height_filter=8, stride=4, zero_padding=0, padding_value=1))
    net.add(BatchNorm())
    net.add(ReLU())
    net.add(ConvolutionLayer(n_filter=64, width_filter=4, height_filter=5, stride=1, zero_padding=0, padding_value=1))
    net.add(BatchNorm())
    net.add(ReLU())
    net.add(MaxPoolLayer(width_filter=2, height_filter=2, stride=2))
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
    net.add(FullyConnectedLayer(1024))
    net.add(BatchNorm())
    net.add(ReLU())
    net.add(Dropout(0.5))
    net.add(FullyConnectedLayer(2))
    net.add(SoftMax())

    optimizer = Adam(learning_rate=0.05)
    net.regression(optimizer=optimizer, cost=CrossEntropyCost())
    net.load('network.h5')

    net.fit_generator(generator=gen, save_step=100, snapshot_step=200)
