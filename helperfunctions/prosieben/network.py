from deepnet import Network
import cv2
import numpy as np
from helperfunctions.image_processing.ads_generator import AdsGenerator
from helperfunctions.image_processing.retrieving_images import VideoCapture
from deepnet.layers import FullyConnectedLayer, BatchNorm, Dropout, ReLU, SoftMax, ConvolutionLayer, MaxPoolLayer, Flatten
from deepnet.optimizers import Adam, SGD
import time
import deepdish as dd

if __name__ == "__main__":

    gen = AdsGenerator(epochs=1, mini_batch_size=64)
    net = Network()

    net.use_gpu = True

    net.input((1, 144, 276))

    net.add(ConvolutionLayer(n_filter=16, width_filter=12, height_filter=8, stride=4))
    net.add(BatchNorm())
    net.add(ReLU())
    net.add(ConvolutionLayer(n_filter=64, width_filter=6, height_filter=4, stride=1))
    net.add(BatchNorm())
    net.add(ReLU())
    net.add(MaxPoolLayer(width_filter=4, height_filter=4, stride=2))
    net.add(ConvolutionLayer(n_filter=128, width_filter=4, height_filter=4, stride=1))
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

    optimizer = Adam(learning_rate=0.001)
    net.regression(optimizer=optimizer, cost="cross_entropy")
    net.fit_generator(generator=gen, save_step=100, snapshot_step=100)
