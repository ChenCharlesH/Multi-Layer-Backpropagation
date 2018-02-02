import neural_util as ut
import numpy as np
import two_layer_nn as tl
import data as dat
import matplotlib.pyplot as plt


def main():
    classes = [x for x in range(0, 10)]

    # Load the training img and labels
    train_images, train_labels = dat.getTrainingData(classes, classes, 0, None)
    # train_images = train_images[0:100,:]
    # train_labels = train_labels[0:100]

    # Load the testing img and labels
    test_images, test_labels = dat.getTestingData(classes, classes, 0, None)
    # test_images = test_images[0:100,:]
    # test_labels = test_labels[0:100]

    # # Perform our input normalization.
    train_images = ut.mean_center_pixel(train_images)
    train_images, vec = ut.kl_expansion_equal_cov(train_images)
    test_images = ut.mean_center_pixel(test_images)
    train_images = train_images
    test_images = np.dot(vec.T, test_images.T).T
    test_images = test_images

    # initiate 2 layer Neural Network with Softmax outputs and Logistic hidden layer
    nn = tl.TwoLayerNN(train_images.shape[1], 64, 10, isTanH = False, isRelu=True, normWeights = True)

    error1 = nn.train(train_images, train_labels, test_images, test_labels, iter=100, n0=.01, T=20, minibatch=128,
    earlyStop=3, reg=0, regNorm = 2, alpha=0.9, isPlot = True, isNumerical = False, isShuffle = True, isNesterov=True)
    # reg = 0.0000001 for L2

    # nn2 = tl.TwoLayerNN(train_images.shape[1], 64, 10, isTanH = True, normWeights = True)
    #
    # # Train the Neural Networ
    # error2 = nn2.train(train_images, train_labels, test_images, test_labels, iter=100, n0=.01, T=100, minibatch=128,
    # earlyStop=3, reg=0.0001, regNorm = 2, alpha=0.9, isPlot = True, isNumerical = False, isShuffle = True)
    #
    # plt.plot(1-error1,label = 'Without Momentum',linewidth=0.8)
    # plt.plot(1-error2, label = 'With Momentum',linewidth=0.8)
    # plt.xlabel("Epochs")
    # plt.ylabel("Accuracy")
    # plt.legend()
    # plt.show()

    # Test the Neural Network
    print "Error Rate: " + str(100 * nn.test(test_images,test_labels)) + str("%")

if __name__ == '__main__':
    main()
