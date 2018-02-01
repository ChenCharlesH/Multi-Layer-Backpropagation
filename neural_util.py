import numpy as np
import matplotlib.pyplot as plt
import math

from random import randint

EXP_CLIP = 100

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

# Separate images and labels into a training set and a holdout set
def get_holdout(images, labels, fraction):
    if(fraction>1):
        fraction = float(fraction)/images.shape[0]
    s = images.shape[0]
    randomVal = np.random.rand(s)
    idx = randomVal<=fraction
    holdout_images = images[idx]
    holdout_labels = labels[idx]
    idx = randomVal>fraction
    train_images = images[idx]
    train_labels = labels[idx]
    return train_images, train_labels, holdout_images, holdout_labels

# 1-pad input data
def pad_ones(images):
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
    for x in range(0, T.size):
            res += T[x] * math.log(Y[x]) + ((1 - T[x]) * math.log(1-Y[x]))
    return -res

# k_cross_entropy
def k_entropy(Y, T):
    res = 0
    rows = Y.shape[0]
    columns = Y.shape[1]
    for i in range(0, rows):
        for j in range(0, columns):
            # if 0 for some reason, set it really close
            if Y[i,j] == 0.0:
                Y[i,j] = 0.00001
            res += T[i,j] * math.log(Y[i,j])
    return res

def k_avg_entropy(Y, T):
    return k_entropy(Y, T) / (Y.shape[0] * T.shape[0])

def avg_cross_entropy(T, Y):
    return cross_entropy(T, Y) / T.shape[0]

# Return error rate of softmax.
def error_rate(res, givenLabel):
    err = 0
    labels1 = np.argmax(res,axis=1)
    labels2 = np.argmax(givenLabel,axis=1)
    for x in range(0, len(labels2)):
        if labels1[x] != labels2[x]:
            err += 1
    return ((float)(err)) / len(givenLabel)

# Returns one hot encoding of images.
def one_hot_encoding(labels):
    res = np.zeros((labels.shape[0],10))
    for i in range(0,len(labels)):
        res[i,int(labels[i])] = 1
    return res

# Returns the gradient for L1 normalization
def l1_grad(x):
    return np.sign(x)

# Returns the gradient for L2 normalization
def l2_grad(x):
    return 2*x

# Logistic activation function
def logistic(x):
    return 1 / (1 + np.exp(-np.clip(x, -EXP_CLIP, EXP_CLIP)))

# Special Tanh activation
def stanh(x):
    return 1.7159 * np.tanh((2.0 / 3.0) * x) 

# Softmax activation function
def softmax(x):
    res = np.exp(-np.clip(x, -EXP_CLIP, EXP_CLIP))
    res_sum = res.sum(axis=1)
    res = res / res_sum[:, np.newaxis]
    return res

def batch(train_images, train_labels, b, minibatch):
    rows = train_images.shape[0]
    start = b*minibatch
    end = (b+1)*minibatch if ((b+1) * minibatch<rows) else rows
    batch_images = train_images[start:end]
    batch_labels = train_labels[start:end]
    return batch_images, batch_labels

# Government z-score
def zscore(images):
    return (images/127.5)-1

# Shuffle the labels and images
def permute(images, labels):
    times = labels.shape[0]
    for i in range(0, times * 2):
        v_i = randint(0, times - 1)
        v_j = randint(0, times - 1)

        images[[v_i, v_j]] = images[[v_j, v_i]]
        labels[[v_i, v_j]] = labels[[v_j, v_i]]

# Get mean of each pixel
def mean_center_pixel(images):
    m = images.mean(axis=0,keepdims=True, dtype=np.float64)
    return images - m

# Do expansion
def kl_expansion_equal_cov(images):
    covm = np.cov(images, rowvar=False)
    val, vec = np.linalg.eig(covm)
    val = np.sqrt(np.absolute(np.real(val))).reshape(1, images.shape[1])
    vec = np.real(vec * val)
    klt = np.dot(vec.T, images.T).T
    return klt, vec
