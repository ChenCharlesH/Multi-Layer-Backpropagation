# File to hold all data preperation logistics.

import numpy as np
import neural_util as ut

from mnist import MNIST

# Line might be OS dependent, change based on OS.
mndata = MNIST('MNIST/')
mndata.gz = True

# Get the data with given classes.
def getTrainingData(numbers, assign, start, end):
    # Only get subset of images and select a subset
    train_images, train_labels = mndata.load_training()
    if end == None:
        train_images = train_images[start:]   
        train_labels = train_labels[start:] 
    else:
        train_images = train_images[start:end]   
        train_labels = train_labels[start:end] 
    train_images, train_labels = ut.getTT(train_images, train_labels, numbers, assign)
    return train_images, train_labels

# Get the data with given classes.
def getTestingData(numbers, assign, start, end):
    # Only get subset of images and select a subset
    test_images, test_labels = mndata.load_testing()
    if end == None:
        test_images = test_images[start:]   
        test_labels = test_labels[start:] 
    else:
        test_images = test_images[start:end]   
        test_labels = test_labels[start:end] 
    test_images, test_labels = ut.getTT(test_images, test_labels, numbers, assign)
    return test_images, test_labels

