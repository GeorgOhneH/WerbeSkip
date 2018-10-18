from deepnet import Network
from helperfunctions.image_processing.image_loader import load_ads_cnn
from deepnet.layers import FullyConnectedLayer, BatchNorm, Dropout, ReLU, SoftMax, ConvolutionLayer, MaxPoolLayer, Flatten
from deepnet.optimizers import Adam
from deepnet.functions.costs import QuadraticCost, CrossEntropyCost

if __name__ == "__main__":
    v_x, v_y, t_x, t_y = load_ads_cnn(split=0, colour=False)

    net = Network()

    net.use_gpu = True

    net.input((1, 52, 52))

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
    net.load("10paddingNoColour.h5")
    # trained with 688000 images
    net.evaluate(t_x, t_y)
    # Evaluation with 7830 data:
    # loss: 0.03852 | accuracy: 0.98966 | precision: 0.99912 | recall: 0.98675 | f1_score: 0.99289
