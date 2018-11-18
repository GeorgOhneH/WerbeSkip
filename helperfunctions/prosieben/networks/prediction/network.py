from deepnet import Network
from deepnet.layers import FullyConnectedLayer, BatchNorm, Dropout, ReLU, SoftMax, ConvolutionLayer, MaxPoolLayer, Flatten
from deepnet.optimizers import Adam
from helperfunctions.image_processing.prediction import visualize_prediction
from deepnet.functions.costs import QuadraticCost, CrossEntropyCost

import cv2
import numpy as np
import os

if __name__ == "__main__":
    directory = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    imgs = [
        "images/zattoo/classified/logo/2000-01-01 01-21-30.jpg",
    ]
    result = []
    for img_name in imgs:
        path = os.path.join(directory, img_name)
        img = cv2.imread(path)
        img = np.transpose(img, (2, 0, 1)).astype(dtype="float16") / 255
        img = np.expand_dims(img, axis=0)
        result.append(img)
    images = np.concatenate(result)

    # Network from example1
    net = Network()

    net.use_gpu = True

    net.input((3, 52, 52))

    net.add(ConvolutionLayer(n_filter=64, width_filter=4, height_filter=4, stride=2, zero_padding=0))
    net.add(BatchNorm())
    net.add(ReLU())
    net.add(ConvolutionLayer(n_filter=128, width_filter=4, height_filter=4, stride=1, zero_padding=0))
    net.add(BatchNorm())
    net.add(ReLU())
    net.add(MaxPoolLayer(width_filter=2, height_filter=2, stride=1))
    net.add(ConvolutionLayer(n_filter=256, width_filter=4, height_filter=4, stride=1, zero_padding=0))
    net.add(BatchNorm())
    net.add(Dropout(0.75))
    net.add(Flatten())
    net.add(FullyConnectedLayer(256))
    net.add(BatchNorm())
    net.add(ReLU())
    net.add(Dropout(0.5))
    net.add(FullyConnectedLayer(2))
    net.add(SoftMax())

    optimizer = Adam(learning_rate=0.001)
    net.regression(optimizer=optimizer, cost=CrossEntropyCost())
    net.load("../example1/parameters.h5")
    visualize_prediction(network=net, images=images, logo_w=52, logo_h=52, stride=2)  # I use stride=2, because I don't have enough ram
