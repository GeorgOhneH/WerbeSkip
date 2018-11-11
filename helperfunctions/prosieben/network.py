from deepnet import Network
import cv2
import numpy as np
from helperfunctions.image_processing.logo_generator import LogoGenerator
from helperfunctions.image_processing.retrieving_images import VideoCapture
from deepnet.layers import FullyConnectedLayer, BatchNorm, Dropout, ReLU, SoftMax, ConvolutionLayer, MaxPoolLayer, Flatten
from deepnet.optimizers import Adam, SGD
from helperfunctions.image_processing.image_loader import load_ads_cnn
from deepnet.functions.costs import CrossEntropyCost
import time
import deepdish as dd

if __name__ == "__main__":
    gen = LogoGenerator(epochs=1, mini_batch_size=1, padding_w=151.5, padding_h=84.5, colour=True, channel="teleboy")
    v_x, v_y, t_x, t_y = load_ads_cnn(split=1, full=True, volume=0.1, colour=True)

    net = Network()

    net.use_gpu = True

    net.input((3, 180, 320))

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
    net.load('network.h5')
    net.print_infos()
    net.fit_generator(generator=gen, save_step=100, snapshot_step=300, validation_set=(v_x, v_y))
