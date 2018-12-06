from deepnet import Network
import cv2
import numpy as np
from helperfunctions.image_processing.ads_generator import AdsGenerator
from helperfunctions.image_processing.image_loader import load_ads_cnn
from helperfunctions.image_processing.video_capture import VideoCapture
from deepnet.layers import FullyConnectedLayer, BatchNorm, Dropout, ReLU, SoftMax, ConvolutionLayer, MaxPoolLayer, Flatten
from deepnet.optimizers import Adam, SGD
from deepnet.functions.costs import CrossEntropyCost
import time
import deepdish as dd

if __name__ == "__main__":
    v_x, v_y, t_x, t_y = load_ads_cnn(split=1, full=True, cropped=True, colour=False)

    # gen = AdsGenerator(epochs=1, mini_batch_size=64, ffmpeg_log="error")
    net = Network()

    net.use_gpu = True

    net.input((1, 141, 269))

    net.add(ConvolutionLayer(n_filter=16, width_filter=9, height_filter=9, stride=4))
    net.add(BatchNorm())
    net.add(ReLU())
    net.add(ConvolutionLayer(n_filter=64, width_filter=6, height_filter=6, stride=1))
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
    net.add(Dropout(0.75))
    net.add(Flatten())
    net.add(FullyConnectedLayer(2048))
    net.add(BatchNorm())
    net.add(ReLU())
    net.add(Dropout(0.5))
    net.add(FullyConnectedLayer(2))
    net.add(SoftMax())

    optimizer = Adam(learning_rate=0.001)
    net.regression(optimizer=optimizer, cost=CrossEntropyCost())
    # net.load("network.h5")
    net.load("random.h5")
    net.print_infos()
    # Number of inputs: 339200, Training time: 156318.02081632614
    net.evaluate(v_x, v_y)
    # Evaluation of 8230 inputs:
    # loss: 1.35391 | accuracy: 0.37169 | MCC: 0.16378

    # net.fit_generator(generator=gen, save_step=100, snapshot_step=100, validation_set=(v_x, v_y))
