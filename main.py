import neural_util as ut
import numpy as np
import two_layer_nn as tl
import data as dat


def main():
    classes = [x for x in range(0, 10)]

    # Load the training img and labels
    train_images, train_labels = dat.getTrainingData(classes, classes, 0, None)

    # Load the testing img and labels
    test_images, test_labels = dat.getTestingData(classes, classes, 0, None)

    # initiate 2 layer Neural Network with Softmax outputs and Logistic hidden layer
    nn = tl.TwoLayerNN(train_images.shape[1], 64, 10)

    # Train the Neural Network
    nn.train(train_images, train_labels, test_images, test_labels, iter=100, n0=0.01,T=100, minibatch=128,
    earlyStop=3, reg=0.0001, regNorm = 2, isPlot = True, isNumerical = False)

    # Test the Neural Network
    print "Error Rate: " + str(100 * nn.test(test_images,test_labels)) + str("%")

if __name__ == '__main__':
    main()
