import numpy as np
import neural_util as ut
import math 

# File to hold softmax.

# Class to house logistical regression.
class SoftMax:
        output = []
        W = np.array([])
        classes = []

        # Constructor
        def __init__(self, dim, classAssign):
            self.DIM = dim
            tempW = []
            for c in classAssign:
                tempW.append([0 for x in range(0,dim)])
                self.output.append([])
            self.W = np.array(tempW)

            self.classes = classAssign
        
        def run(self, dataM):
            res = np.matmul(dataM, self.W);
            res = np.exp(np.clip(res,-100,100))
            res_sum = res.sum(axis=1)
            res = res / res_sum[:, np.newaxis]
            return res

        # Get the probability of the largest
        def run2(self, dataM):
            A = []
            res = []
            # calculate each ak
            for c in range(0, len(self.classes)):
                A.append(np.dot(dataM, self.W[c]))
            

            # Calculate the exps
            for dRowCnt in range(0, dataM.shape[0]):
                # Bottom total
                denom = 0.0
                for c in range(0, len(self.classes)):
                    denom += math.exp(np.clip(A[c][dRowCnt], -500, 500))
                
                m = []
                for c in range(0, len(self.classes)):
                    r = math.exp(np.clip(A[c][dRowCnt], -500, 500)) / denom
                    m.append(r)
                
                res.append(self.classes[np.argmax(m)])

            return np.array(res)
        
        # Runs the softmax node and output results
        # Returns a vector of probability of each class.
        def run_all(self, dataM):
            A = []
            
            self.output = []
            for c in self.classes:
                self.output.append([])

            # calculate each ak
            for c in range(0, len(self.classes)):
                A.append(np.dot(dataM, self.W[c]))

            # Calculate the exps
            for dRowCnt in range(0, dataM.shape[0]):
                # Bottom total
                denom = 0.0
                for c in range(0, len(self.classes)):
                    denom += math.exp(np.clip(A[c][dRowCnt], -500, 500))

                for c in range(0, len(self.classes)):
                    r = math.exp(np.clip(A[c][dRowCnt], -500, 500)) / denom
                    self.output[c].append(r)

            return np.array(self.output)
            
        