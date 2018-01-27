import neural_util as ut
import numpy as np
import soft_gradient_descent as sd
import softmax_regression as sr
import data as dat


def main():
    classes = [x for x in range(0, 10)]

    # Load the training img and labels
    train_images, train_labels = dat.getTrainingData(classes, classes, 0, None)
    #train_images = train_images[0:100,:]
    #train_labels = train_labels[0:100]

    # Load the testing img and labels
    test_images, test_labels = dat.getTestingData(classes, classes, 0, None)
    # test_images = test_images[0:100,:]
    # test_labels = test_labels[0:100]

    # initiate 2 layer Neural Network with Softmax outputs and Logistic hidden layer
    nn = sr.TwoLayerNN(train_images.shape[1], 64, 10)

    # Train the Neural Network
    nn.train(train_images, train_labels, iter=100, n0=0.001,T=100, minibatch=128,
    earlyStop=3, reg=0.0001, regNorm = 2, isPlot = False)

    # Test the Neural Network
    print "Error Rate: " + str(100 * nn.test(test_images,test_labels)) + str("%")

if __name__ == '__main__':
    main()
