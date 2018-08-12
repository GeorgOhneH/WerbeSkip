from deepnet import Network
from app.image_processing.image_loader import load_ads_cnn
from deepnet.layers import FullyConnectedLayer, BatchNorm, Dropout, ReLU, SoftMax, ConvolutionLayer, MaxPoolLayer, Flatten
from deepnet.optimizers import Adam

if __name__ == "__main__":
    v_x, v_y, t_x, t_y = load_ads_cnn(split=0, colour=True)

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
    net.regression(optimizer=optimizer, cost="cross_entropy")
    net.load("10padding.h5")
    # Trained with 976000 images
    net.evaluate(t_x, t_y)
    # loss: 0.05221 | accuracy: 0.98480 | precision: 0.99646 | recall: 0.98274 | f1_score: 0.98955
