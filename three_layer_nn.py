import numpy as np
import neural_util as ut
import matplotlib.pyplot as plt

# Class to house logistical regression.
class ThreeLayerNN:

    # Weights should be column vectors
    W1 = np.array([])
    W2 = np.array([])
    W3 = np.array([])

    grad1 = np.array([])
    grad2 = np.array([])
    grad3 = np.array([])

    X = np.array([])
    Z1 = np.array([])
    Z2 = np.array([])
    Y = np.array([])
    isTanH = False

    # Constructor
    def __init__(self, n1, n2, n3, n4, isTanH = False, normWeights = False):
        # TODO: Randomize the starting weights with N(0,sigma^2)
        if normWeights:
            self.W1 = np.random.normal(0,1.0/np.sqrt(n1+1),size = (n1 + 1, n2))
            self.W2 = np.random.normal(0,1.0/np.sqrt(n2+1),size = (n2 + 1, n3))
            self.W3 = np.random.normal(0,1.0/np.sqrt(n3+1),size = (n3 + 1, n4))
        else:
            self.W1 = np.random.normal(0,0.1,size = (n1 + 1, n2))
            self.W2 = np.random.normal(0,0.1,size = (n2 + 1, n3))
            self.W2 = np.random.normal(0,0.1,size = (n3 + 1, n4))
        self.grad1 = np.zeros(self.W1.shape)
        self.grad2 = np.zeros(self.W2.shape)
        self.grad3 = np.zeros(self.W3.shape)
        self.isTanH = isTanH

    def run(self, data):
        # Hidden Layer
        self.X = np.copy(data)

        Aj1 = np.matmul(self.X, self.W1)
        if self.isTanH:
            self.Z1 = ut.stanh(Aj1)
        else:
            self.Z1 = ut.logistic(Aj1)
        self.Z1 = ut.pad_ones(self.Z1)

        Aj2 = np.matmul(self.Z1, self.W2)
        if self.isTanH:
            self.Z2 = ut.stanh(Aj2)
        else:
            self.Z2 = ut.logistic(Aj2)
        self.Z2 = ut.pad_ones(self.Z2)

        # Output Layer
        Ak = np.matmul(self.Z2, self.W3)
        self.Y = ut.softmax(Ak)

        return self.Y

    # Backpropagation step for given images, labels, and learning rate n
    # We can normalize our batch
    # Returns the two weights.
    def backprop(self, batch_images, batch_labels, n, alpha, reg, regNorm, isNumerical = False):
        #TODO add reg, regNorm
        delta3 = np.subtract(self.run(batch_images), batch_labels)
        if self.isTanH:
            delta2 = np.multiply( (2.0/3.0)/1.7159 * np.multiply(1.7159 - self.Z2[:, 1:],1.7159 + self.Z2[:, 1:]) ,np.matmul(delta3, self.W3.T[:, 1:]))
        else:
            delta2 = np.multiply(np.multiply(self.Z2[:, 1:],(1 - self.Z2[:, 1:])),np.matmul(delta3, self.W3.T[:,1:]))
        if self.isTanH:
            delta1 = np.multiply( (2.0/3.0)/1.7159 * np.multiply(1.7159 - self.Z1[:, 1:],1.7159 + self.Z1[:, 1:]) ,np.matmul(delta2, self.W2.T[:, 1:]))
        else:
            delta1 = np.multiply(np.multiply(self.Z1[:, 1:],(1 - self.Z1[:, 1:])),np.matmul(delta2, self.W2.T[:,1:]))
        grad3 = np.matmul(np.transpose(self.Z2),delta3)
        grad2 = np.matmul(np.transpose(self.Z1),delta2)
        grad1 = np.matmul(np.transpose(self.X),delta1)
        # if isNumerical:
        #     agrad1, agrad2 = self.numApprox(batch_images, batch_labels, 0.01)
        #     print "Grad1:"
        #     print grad1[0:10,0:5]
        #     print agrad1[0:10,0:5]
        #     print "---------------------------"
        #     print "Grad2:"
        #     print grad2[0:10,0:5]
        #     print agrad2[0:10,0:5]
        #     print "---------------------------"
        #     print "Max Difference Input Bias:"
        #     print np.max(np.absolute(np.subtract(grad1[0,0:5], agrad1[0,0:5])))
        #     print "Max Difference Output Bias:"
        #     print np.max(np.absolute(np.subtract(grad2[0,0:5], agrad2[0,0:5])))
        #     print "Max Difference First Layer Weights:"
        #     print np.max(np.absolute(np.subtract(grad1[1:10,0:5], agrad1[1:10,0:5])))
        #     print "Max Difference Second Layer Weights:"
        #     print np.max(np.absolute(np.subtract(grad2[1:10,0:5], agrad2[1:10,0:5])))
        #     print "---------------------------"
        self.grad3 = alpha*self.grad3 + n*grad3/batch_labels.shape[0]
        self.grad2 = alpha*self.grad2 + n*grad2/batch_labels.shape[0]
        self.grad1 = alpha*self.grad1 + n*grad1/batch_labels.shape[0]
        self.W3 = self.W3 + self.grad3
        self.W2 = self.W2 + self.grad2
        self.W1 = self.W1 + self.grad1

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

        grad3 = np.zeros(self.W3.shape)
        for i in range(0,10):
            for j in range(0,5):
                self.W3[i,j] = self.W3[i,j] + epsilon
                Ypos = self.run(images)
                self.W3[i,j] = self.W3[i,j] - 2*epsilon
                Yneg = self.run(images)
                self.W3[i,j] = self.W3[i,j] + epsilon
                grad2[i,j] = (ut.k_entropy(Ypos,labels)-ut.k_entropy(Yneg,labels))/(2*epsilon)

        return grad1,grad2,grad3

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
        minW3 = self.W3
        train_images, train_labels, holdout_images, holdout_labels = ut.get_holdout(train_images, train_labels,0.1)
        train_images = ut.zscore(train_images)
        train_images = ut.pad_ones(train_images)
        train_labels = ut.one_hot_encoding(train_labels)

        # Collection data attributes
        errorTrain = []
        errorHoldout = []
        errorTest = []
        testLoss = []
        holdoutLoss = []
        trainLoss = []

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

                testLoss.append(self.k_entropy(train_images, train_labels, False))
                holdoutLoss.append(self.k_entropy(holdout_images, holdout_labels))
                trainLoss.append(self.k_entropy(test_images, test_labels))

            print errorNew

            # Keep track of Best performance
            if errorNew < minError:
                minError = errorNew
                minW1 = np.copy(self.W1)
                minW2 = np.copy(self.W2)
                minW3 = np.copy(self.W3)

            if errorNew > errorOld:
                stopCount = stopCount + 1
                if stopCount==earlyStop and t>minIter:
                    break
            else:
                stopCount = 0

        self.W1 = minW1
        self.W2 = minW2
        self.W3 = minW3

        if isPlot:
            plt.plot(1-np.array(errorTrain),label = 'Training',linewidth=0.8)
            plt.plot(1-np.array(errorHoldout), label = 'Holdout',linewidth=0.8)
            plt.plot(1-np.array(errorTest), label = 'Test',linewidth=0.8)
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.show()

            plt.plot(np.array(trainLoss),label = 'Training',linewidth=0.8)
            plt.plot(np.array(holdoutLoss), label = 'Holdout',linewidth=0.8)
            plt.plot(np.array(testLoss), label = 'Test',linewidth=0.8)
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()
        return np.array(errorTrain)

    def k_entropy(self, test_images, test_labels, format=True):
        if format:
            test_images = ut.zscore(test_images)
            test_images = ut.pad_ones(test_images)
            test_labels = ut.one_hot_encoding(test_labels)
        return ut.k_entropy(self.run(test_images), test_labels)

    def test(self, test_images, test_labels, format=True):
        if format:
            test_images = ut.zscore(test_images)
            test_images = ut.pad_ones(test_images)
            test_labels = ut.one_hot_encoding(test_labels)
        return ut.error_rate(self.run(test_images), test_labels)
