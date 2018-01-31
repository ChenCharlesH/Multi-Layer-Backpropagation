import numpy as np
import neural_util as ut

# Class to house logistical regression.
class TwoLayerNN:
    output = []
    # Weights should be column vectors.
    W1 = np.array([])
    # Weights should be column vectors
    W2 = np.array([])

    X1 = np.array([])
    X2 = np.array([])


    # Constructor
    def __init__(self, n1, n2, n3):
        # TODO: Randomize the starting weights
        # data is of size N*n1
        self.W1 = np.zeros(shape = (n1 + 1, n2))
        self.W2 = np.zeros(shape = (n2 + 1, n3))

    def run(self, data):
        # Hidden Layer Input
        self.X1 = np.copy(data)
        res = np.matmul(data, self.W1)

        # Hidden Layer Output
        res = ut.logistic(res)


        # Output Layer Input
        res = ut.pad_ones(res)
        self.X2 = np.copy(res)
        res = np.matmul(res, self.W2)

        # Output Layer Output
        res = ut.softmax(res)

        return res

    # Backpropagation step for given images, labels, and learning rate n
    # We can normalize our batch
    # Returns the two weights.
    def backprop(self, batch_images, batch_labels, n, reg, regNorm):
        #TODO add reg, regNorm
        #TODO maybe add normalization of gradient w.r.t. size of batch
        # Hidden to Output
        # n x 10
        delta2 = np.subtract(self.run(batch_images), batch_labels)
        # n x 65
        z2 = self.X2

        # Input to Hidden
        # Since this is only one layer, we can simply, matmul batch_images and W1
        # to get a1.
        # n x 785
        z1 = self.X1
        # n x 65
        sum_term = np.matmul(delta2, self.W2.T[:, :-1])
        # Note the + -1 is necessary

        # n x 65
        delta1 = np.multiply(np.multiply(z2[:, 1:], (z2[:, 1:] - 1)), sum_term)

        # Update weights
        dir_res2 = (n * np.matmul(z2.T, delta2)) / batch_labels.shape[0]
        dir_res1 = (n * np.matmul(z1.T, delta1)) / batch_labels.shape[0]

        res2 = np.add(self.W2, dir_res2)
        res1 = np.add(self.W1, dir_res1)

        return res1, res2, dir_res1, dir_res2

    # Helper for numApprox
    def assigner(self, x, W):
        if x == 0:
            self.W1 = W
        else:
            self.W2 = W

    # Function to find numerical approximation.
    def numApprox(self, images, labels, epsilon):
        values = [self.W1, self.W2]
        res = []
        for t in range(0, len(values)):
            W = values[t]
            prev = np.copy(W)


            # Generate three grids
            eGrid = np.zeros(shape=W.shape)
            aGrid = np.zeros(shape=W.shape)
            sGrid = np.zeros(shape=W.shape)
            eGrid.fill(epsilon)

            aW = np.add(W, eGrid)
            sW = np.subtract(W, eGrid)

            # Calculate + and - values
            for i in range(0, W.shape[0]):
                for j in range(0, W.shape[1]):
                    # Set forward
                    W[i, j] = aW[i, j]
                    # Set specific weights
                    self.assigner(t, W)
                    aGrid[i, j] = self.test(images, labels, False)

                    W[i, j] = sW[i, j]
                    self.assigner(t, W)
                    sGrid[i, j] = self.test(images, labels, False)

                    # set values back.
                    self.assigner(t, prev)

            # calculate our gradient values.
            r = np.subtract(aGrid, sGrid) / (2 * epsilon)
            res.append(r)
            print str(t) + " DONE"
        return res
                    


    def train(self, train_images, train_labels, iter=100, n0=0.001, T=100, minibatch=128, earlyStop=3, minIter=10, reg=0.0001, regNorm = 2, isPlot = False):
        #TODO: Plot the outputs for isPlot
        stopCount = 0
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

                # Returns weights and also the derivatives
                bw1, bw2, dir1, dir2 = self.backprop(batch_images, batch_labels, n, reg, regNorm)
                nA = self.numApprox(batch_images, batch_labels, 0.01)
                print dir1
                print nA[0]
                print "---------------------------"
                print dir2
                print nA[1]
                print "---------------------------"
                print np.subtract(dir1, nA[0])
                print np.subtract(dir2, nA[1])
                print "---------------------------"

                self.W1 = bw1
                self.W2 = bw2

            errorNew = self.test(holdout_images,holdout_labels)

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

    def test(self, test_images, test_labels, format=True):
        if format:
            test_images = ut.zscore(test_images)
            test_images = ut.pad_ones(test_images)
            test_labels = ut.one_hot_encoding(test_labels)
        return ut.error_rate(self.run(test_images), test_labels)
