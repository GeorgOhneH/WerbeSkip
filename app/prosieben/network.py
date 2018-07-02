from deepnet import Network
from app.image_processing.image_loader import load_ads_cnn
from app.image_processing.generator import TrainGenerator
from deepnet.layers import FullyConnectedLayer, BatchNorm, Dropout, ReLU, SoftMax, ConvolutionLayer, MaxPoolLayer, Flatten
from deepnet.optimizers import Adam, SGD

if __name__ == "__main__":

    generator = TrainGenerator(epochs=1, mini_batch_size=16, padding_w=1, padding_h=1, n_workers=1, channel="teleboy")

    net = Network()

    net.use_gpu = True

    net.input((3, 67, 41))

    net.add(ConvolutionLayer(n_filter=64, width_filter=3, height_filter=3, stride=2, zero_padding=0))
    net.add(BatchNorm())
    net.add(ReLU())
    net.add(ConvolutionLayer(n_filter=128, width_filter=4, height_filter=4, stride=1, zero_padding=0))
    net.add(BatchNorm())
    net.add(ReLU())
    net.add(MaxPoolLayer(width_filter=2, height_filter=2, stride=1))
    net.add(ConvolutionLayer(n_filter=256, width_filter=4, height_filter=5, stride=2, zero_padding=0))
    net.add(BatchNorm())
    net.add(ReLU())
    net.add(ConvolutionLayer(n_filter=512, width_filter=4, height_filter=4, stride=1, zero_padding=0))
    net.add(BatchNorm())
    net.add(ReLU())
    net.add(Dropout(0.75))
    net.add(Flatten())
    net.add(FullyConnectedLayer(256))
    net.add(BatchNorm())
    net.add(ReLU())
    net.add(Dropout(0.5))
    net.add(FullyConnectedLayer(2))
    net.add(SoftMax())

    optimizer = Adam(learning_rate=0.001)
    net.regression(optimizer=optimizer, cost="cross_entropy")
    net.fit_generator(generator=generator)
