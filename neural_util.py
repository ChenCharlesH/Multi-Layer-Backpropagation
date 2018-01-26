import numpy as np
import matplotlib.pyplot as plt
import math

# File to hold all helper functions.

# Calulate regularization const for L1

# Create one hot
def toOneHot(cat, size):
    res = []
    for c in range(0, size):
        if c == cat:
           res.append(1)
        else:
           res.append(0)
    
    return np.array(res)

# Convert a function from onehot to category.
def toCat(oneHot):
    for c in range(0, oneHot.size):
        if oneHot[c] == 1:
            return c
    return -1

# Get numbers in list
# labelAs given corrspond to what labels should be assigned
def getTT(images, labels, numbers = [2, 3], labelAs=[1, 0]):
    resX = []
    resY = []
    for x in range(0, len(images)):
        for i in range(0, len(numbers)):
            if labels[x] == numbers[i]:
                resX.append(images[x])
                resY.append(labelAs[i])
                break
    return np.array(resX), np.array(resY)

# Get subset of images belonging to the array passed
def getSubset(images, labels, values):
    resX = []
    resY = []
    for i in range(0, len(values)):
        mask = labels==values[i]
        resX.append(images[mask])
        resY.append(values[i])
    return np.array(resX), np.array(resY)

def getHoldout(images, labels, fraction):
    s = images.shape[0]
    randomVal = np.random.rand(s)
    idx = randomVal<=fraction
    holdout_images = images[idx]
    holdout_labels = labels[idx]
    idx = randomVal>fraction
    train_images = images[idx]
    train_labels = labels[idx]
    return train_images, train_labels, holdout_images, holdout_labels

# Splits the labels and images into fractions for divisons.
def getHoldout_OHE(images, labels, ohe_labels, fraction):
    s = images.shape[0]
    randomVal = np.random.rand(s)
    idx = randomVal<=fraction
    holdout_images = images[idx]
    holdout_labels = labels[idx]
    holdout_ohe_labels = ohe_labels[idx, :]
    idx = randomVal>fraction
    train_images = images[idx]
    train_labels = labels[idx]
    train_ohe_labels = ohe_labels[idx, :]
    return train_images, train_labels, train_ohe_labels, holdout_images, holdout_labels, holdout_ohe_labels

# 1-pad input data 
def padOnes(images):
	s = images.shape
	res = np.ones(shape=(s[0], s[1] + 1))
	res[:, 1:] = images
	return res

# Plot grayscale image
def showImg(image):
	image = image[1:]
	image = np.array(image, dtype='uint8')
	image = image.reshape((28, 28))
	plt.imshow(image, cmap="gray")
	plt.show()

# Value of cross entropy.
def cross_entropy(Y,T):
    res = 0
    # math.log(min(Y))
    # print math.log(Y.min)
    for x in range(0, T.size):
            res += T[x] * math.log(Y[x]) + ((1 - T[x]) * math.log(1-Y[x]))
    return -res

# k_cross_entropy
def k_entropy(Y, T):
    res = 0
    rows = Y.shape[0]

    for r in range(0, rows):
        for v in range(0, T.shape[0]):
            # if 0 for some reason, set it really close
            if Y[r,v] == 0.0:
                Y[r,v] = 0.00001
            res += T[v,r] * math.log(Y[r,v])
    return res

def k_avg_entropy(Y, T):
    return k_entropy(Y, T) / (Y.shape[0] * T.shape[0])

def avg_cross_entropy(T, Y):
    return cross_entropy(T, Y) / T.shape[0]

# Clipped values due to overflow
def sig(x):
    return 1 / (1 + math.exp(-np.clip(x, -500, 30)))

# gets error rate of result
def error_rate(res, givenLabel):
    err = 0
    for x in range(0, len(res)):
        if res[x] != givenLabel[x]:
            err += 1
    
    return ((float)(err)) / givenLabel.size

# error rate for non-rounded data
def error_rate2(res, givenLabel):
    err = 0
    res = round(res)
    for x in range(0, len(res)):
        if res[x] != givenLabel[x]:
            err += 1
    return ((float)(err)) / givenLabel.size

def error_rate3(res, givenLabel):
    err = 0
    labels1 = np.argmax(res,axis=1)
    labels2 = np.argmax(givenLabel,axis=1)
    for x in range(0, len(labels2)):
        if labels1[x] != labels2[x]:
            err += 1
    return ((float)(err)) / len(givenLabel)

# Custum round function to round by clipping.
def round(res):
    res = np.clip(np.around(res, decimals=0), 0, 1)
    return res

def oneHotEncoding(labels):
    res = np.zeros((labels.shape[0],10))
    for i in range(0,len(labels)):
        res[i,int(labels[i])] = 1
    return res

# Custom function to find derivative of absolute value function
def d_abs(x):
    if x==0:
        return 0
    return abs(x) / x

# Vectorized version of the sigma function.
vect_sig = np.vectorize(sig)

# Vectorized version of the derivative of the absolute value function.
vect_d_abs = np.vectorize(d_abs)