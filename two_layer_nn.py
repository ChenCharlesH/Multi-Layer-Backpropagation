import numpy as np
import neural_util as ut
import matplotlib.pyplot as plt

# Class to house logistical regression.
class TwoLayerNN:

    # Weights should be column vectors
    W1 = np.array([])
    W2 = np.array([])

    X = np.array([])
    Z = np.array([])
    Y = np.array([])
    isTanH = False

    # Constructor
    def __init__(self, n1, n2, n3, isTanH = False):
        # TODO: Randomize the starting weights with N(0,sigma^2)
        # self.W1 = np.random.normal(0,1/(n1+1),size = (n1 + 1, n2))
        # self.W2 = np.random.normal(0,1/(n2+1),size = (n2 + 1, n3))
        self.W1 = np.random.normal(0,0.1,size = (n1 + 1, n2))
        self.W2 = np.random.normal(0,0.1,size = (n2 + 1, n3))
        self.isTanH = isTanH
        # self.W1 = np.random.randn(n1 + 1, n2)
        # self.W2 = np.random.randn(n2 + 1, n3)

    def run(self, data):
        # Hidden Layer
        self.X = np.copy(data)
        Aj = np.matmul(self.X, self.W1)
        if self.isTanH:
            self.Z = ut.stanh(Aj)
        else:
            self.Z = ut.logistic(Aj)
        self.Z = ut.pad_ones(self.Z)

        # Output Layer
        Ak = np.matmul(self.Z, self.W2)
        self.Y = ut.softmax(Ak)

        return self.Y

    # Backpropagation step for given images, labels, and learning rate n
    # We can normalize our batch
    # Returns the two weights.
    def backprop(self, batch_images, batch_labels, n, alpha, reg, regNorm, isNumerical = False):
        #TODO add reg, regNorm
        delta2 = np.subtract(self.run(batch_images), batch_labels)
        if self.isTanH:
            #TODO: Fix the gradient for tanh
            delta1 = np.multiply( (2.0/3.0)/1.7159 * np.multiply(1.7159 - self.Z[:, 1:],1.7159 + self.Z[:, 1:]) ,np.matmul(delta2, self.W2.T[:, 1:]))
        else:
            delta1 = np.multiply(np.multiply(self.Z[:, 1:],(1 - self.Z[:, 1:])),np.matmul(delta2, self.W2.T[:,1:]))
        grad2 = np.matmul(np.transpose(self.Z),delta2)
        grad1 = np.matmul(np.transpose(self.X),delta1)
        if isNumerical:
            agrad1, agrad2 = self.numApprox(batch_images, batch_labels, 0.01)
            print "Grad1:"
            print grad1[0:10,0:5]
            print agrad1[0:10,0:5]
            print "---------------------------"
            print "Grad2:"
            print grad2[0:10,0:5]
            print agrad2[0:10,0:5]
            print "---------------------------"
            print "Max Difference Input Bias:"
            print np.max(np.absolute(np.subtract(grad1[0,0:5], agrad1[0,0:5])))
            print "Max Difference Output Bias:"
            print np.max(np.absolute(np.subtract(grad2[0,0:5], agrad2[0,0:5])))
            print "Max Difference First Layer Weights:"
            print np.max(np.absolute(np.subtract(grad1[1:10,0:5], agrad1[1:10,0:5])))
            print "Max Difference Second Layer Weights:"
            print np.max(np.absolute(np.subtract(grad2[1:10,0:5], agrad2[1:10,0:5])))
            print "---------------------------"
        self.W2 = self.W2 + n*grad2/batch_labels.shape[0]
        self.W1 = self.W1 + n*grad1/batch_labels.shape[0]

    def numApprox(self, images, labels, epsilon):
        grad1 = np.zeros(self.W1.shape)
        for i in range(0,10):
            for j in range(0,5):
                self.W1[i,j] = self.W1[i,j] + epsilon
                Ypos = self.run(images)
                self.W1[i,j] = self.W1[i,j] - 2*epsilon
                Yneg = self.run(images)
                self.W1[i,j] = self.W1[i,j] + epsilon
                grad1[i,j] = (ut.k_entropy(Ypos,labels)-ut.k_entropy(Yneg,labels))/(2*epsilon)

        grad2 = np.zeros(self.W2.shape)
        for i in range(0,10):
            for j in range(0,5):
                self.W2[i,j] = self.W2[i,j] + epsilon
                Ypos = self.run(images)
                self.W2[i,j] = self.W2[i,j] - 2*epsilon
                Yneg = self.run(images)
                self.W2[i,j] = self.W2[i,j] + epsilon
                grad2[i,j] = (ut.k_entropy(Ypos,labels)-ut.k_entropy(Yneg,labels))/(2*epsilon)

        return grad1,grad2

    def train(
        self, train_images, train_labels, test_images, test_labels,
        iter=100, n0=0.001, T=100, minibatch=128, earlyStop=3, minIter=10,
        reg=0.0001, regNorm = 2, alpha = 0.9, isPlot = False, isNumerical = False, isShuffle = True
    ):
        #TODO: Plot the outputs for isPlot
        stopCount = 0
        minError = 1
        minW1 = self.W1
        minW2 = self.W2
        train_images, train_labels, holdout_images, holdout_labels = ut.get_holdout(train_images, train_labels,0.1)
        train_images = ut.zscore(train_images)
        train_images = ut.pad_ones(train_images)
        train_labels = ut.one_hot_encoding(train_labels)

        # Collection data attributes
        errorTrain = []
        errorHoldout = []
        errorTest = []

        for t in range(0,iter):
            #TODO: Randomize the data after every epoch through the data
            n = n0/(1+t/T)
            if isShuffle:
                idx = np.random.shuffle(np.arange(train_images.shape[0]))
                train_images[:,:] = train_images[idx,:]
                train_labels[:,:] = train_labels[idx,:]

            errorOld = self.test(holdout_images,holdout_labels)
            for m in range(0, int(np.ceil(float(train_images.shape[0])/minibatch))):
                batch_images,batch_labels = ut.batch(train_images,train_labels, m, minibatch)
                self.backprop(batch_images, batch_labels, n, alpha, reg, regNorm, isNumerical)

            errorNew = self.test(holdout_images,holdout_labels)
            if isPlot:
                errorTrain.append(self.test(train_images, train_labels, False))
                errorHoldout.append(self.test(holdout_images, holdout_labels))
                errorTest.append(self.test(test_images, test_labels))
            print errorNew

            # Keep track of Best performance
            if errorNew < minError:
                minError = errorNew
                minW1 = self.W1
                minW2 = self.W2

            if errorNew > errorOld:
                stopCount = stopCount + 1
                if stopCount==earlyStop and t>minIter:
                    break
            else:
                stopCount = 0

        self.W1 = minW1
        self.W2 = minW2

        if isPlot:
            plt.plot(errorTrain,label = 'Training',linewidth=0.8)
            plt.plot(errorHoldout, label = 'Holdout',linewidth=0.8)
            plt.plot(errorTest, label = 'Test',linewidth=0.8)
            plt.legend()
            plt.show()
        return np.array(errorTrain)

    def test(self, test_images, test_labels, format=True):
        if format:
            test_images = ut.zscore(test_images)
            test_images = ut.pad_ones(test_images)
            test_labels = ut.one_hot_encoding(test_labels)
        return ut.error_rate(self.run(test_images), test_labels)
