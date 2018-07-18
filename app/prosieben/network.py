from deepnet import Network
import cv2
import numpy as np
from app.image_processing.image_loader import load_ads_cnn
from app.image_processing.generator import TrainGenerator
from app.image_processing.prediction import visualize_prediction
from deepnet.layers import FullyConnectedLayer, BatchNorm, Dropout, ReLU, SoftMax, ConvolutionLayer, MaxPoolLayer, Flatten
from deepnet.optimizers import Adam, SGD

if __name__ == "__main__":

    img = cv2.imread("C:\Jetbrains\PyCharm\WerbeSkip\\app\prosieben\images\zattoo\classified\logo\\2000-01-01 01-22-00.jpg")
    img = np.transpose(img, (2, 0, 1)).astype(dtype="float32") / 255
    img = np.expand_dims(img, axis=0)[:,:,:200, 700:]

    v_x, v_y, t_x, t_y = load_ads_cnn(split=0, padding_w=100, padding_h=100, center=True, cache=True)

    net = Network()

    net.use_gpu = True

    net = Network()

    net.use_gpu = True

    net.input((3, 232, 232))

    net.add(ConvolutionLayer(n_filter=8, width_filter=6, height_filter=6, stride=2, zero_padding=0))
    net.add(BatchNorm())
    net.add(ReLU())
    net.add(ConvolutionLayer(n_filter=16, width_filter=6, height_filter=6, stride=2, zero_padding=0))
    net.add(BatchNorm())
    net.add(ReLU())
    net.add(ConvolutionLayer(n_filter=16, width_filter=5, height_filter=5, stride=2, zero_padding=0))
    net.add(BatchNorm())
    net.add(ReLU())
    net.add(ConvolutionLayer(n_filter=32, width_filter=5, height_filter=5, stride=1, zero_padding=0))
    net.add(BatchNorm())
    net.add(ReLU())
    net.add(Dropout(0.75))
    net.add(Flatten())
    net.add(FullyConnectedLayer(128))
    net.add(ReLU())
    net.add(Dropout(0.5))
    net.add(FullyConnectedLayer(2))
    net.add(SoftMax())

    optimizer = Adam(learning_rate=0.001)
    net.regression(optimizer=optimizer, cost="cross_entropy")
    net.load("100padding.h5")
    net.evaluate(t_x, t_y)
