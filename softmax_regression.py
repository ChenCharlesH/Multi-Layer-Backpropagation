import numpy as np
import neural_util as ut

# Class to house logistical regression.
class TwoLayerNN:
    output = []
    # Weights should be column vectors.
    W1 = np.array([])
    # Weights should be column vectors
    W2 = np.array([])


    # Constructor
    def __init__(self, n1, n2, n3):
        # TODO: Randomize the starting weights
        # data is of size N*n1
        self.W1 = np.zeros(shape = (n1 + 1, n2 + 1))
        self.W2 = np.zeros(shape = (n2 + 1, n2))

    def run(self, data):
        # Hidden Layer Input
        res = np.matmul(data, self.W1)
        # Hidden Layer Output
        res = ut.logistic(res)

        # Output Layer Input
        res = np.matmul(res, self.W2)
        # Output Layer Output
        res = ut.softmax(res)

        return res

    # Backpropagation step for given images, labels, and learning rate n
    # We can normalize our batch
    def backprop(self, batch_images, batch_labels, n, reg, regNorm):
        #TODO add reg, regNorm
        # Hidden to Output
        delta2 = np.sum(np.subtract(self.run(batch_images), batch_labels))

        # Input to Hidden
        # Since this is only one layer, we can simply, matmul batch_images and W1
        # to get a1.
        sig1 = ut.logistic(np.matmul(batch_images, self.W1))
        delta1 = sig1* (1-sig1) * delta2 * self.W1.shape[1] * np.sum(self.W1)

        # Update weights
        self.W2 = np.add(self.W2, np.sum(n  * delta2 * batch_images, axis=0))
        self.W1 = np.add(self.W1, np.sum(n * delta1 * batch_images, axis = 0))

    def train(self, train_images, train_labels, iter=100, n0=0.001, T=100, minibatch=128, earlyStop=3, minIter=10, reg=0.0001, regNorm = 2, isPlot = False):
        #TODO: Plot the outputs for isPlot
        stopCount = 0;
        minError = 1
        minW1 = self.W1
        minW2 = self.W2
        train_images, train_labels, holdout_images, holdout_labels = ut.get_holdout(train_images, train_labels, 0.1)

        train_images = ut.zscore(train_images)
        train_images = ut.pad_ones(train_images)
        train_labels = ut.one_hot_encoding(train_labels)

        for t in range(0,iter):
            #TODO: Randomize the data after every epoch through the data
            n = n0/(1+t/T)

            errorOld = self.test(holdout_images,holdout_labels)
            for m in range(0, int(np.ceil(train_images.shape[0]/minibatch))):
                batch_images,batch_labels = ut.batch(train_images,train_labels, m, minibatch)
                self.backprop(batch_images, batch_labels, n, reg, regNorm)
            errorNew = self.test(holdout_images,holdout_labels)

            # Keep track of Best performance
            if errorNew < minError:
                minError = errorNew
                minW1 = self.W1
                minW2 = self.W2

            # Early Stop Condition
            if errorNew < minError:
                minError = errorNew
                minErrorWeight = W

            if errorNew > errorOld:
                stopCount = stopCount + 1
                if stopCount==earlyStop and t>minIter:
                    break
            else:
                stopCount = 0

        self.W1 = minW1
        self.W2 = minW2

    def test(self, test_images, test_labels):
        test_images = ut.zscore(test_images)
        test_images = ut.pad_ones(test_images)
        test_labels = ut.one_hot_encoding(test_labels)
        return ut.error_rate(self.run(test_images), test_labels)
