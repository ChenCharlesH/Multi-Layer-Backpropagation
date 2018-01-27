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

        

        