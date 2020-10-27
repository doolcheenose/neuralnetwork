import numpy as np
import gzip
import time
from random import random
from PIL import Image

class Network:
    def __init__(self, weights, biases, learning_rate):
        if len(weights) != len(biases):
            raise ValueError('Incompatible weights and biases')
        self.weights = weights
        self.biases = biases
        self.learning_rate = learning_rate

    def trainAndUpdate(self, dataset):
        for (input, target) in dataset:
            z = [np.empty(0) for i in range(len(self.weights) + 1)] # z[0] will be left empty for indexing convenience
            a = [np.empty(0) for i in range(len(self.weights) + 1)]
            a[0] = input.copy()
            for i, weight in enumerate(self.weights):
                z[i+1] = np.dot(weight, a[i]) + self.biases[i]
                a[i+1] = sigma(z[i+1])
            costvec = target - a[-1]

            # backpropagate occurs here
            dCda = [np.empty(0) for i in range(len(self.weights) + 1)]
            dCda[-1] = costvec.copy()
            dCdw = [np.empty(0) for i in range(len(self.weights))]
            dCdw[-1] = np.outer(sigmaprime(z[-1]) * dCda[-1], a[-2])
            dCdb - [np.empty(0) for i in range(len(self.biases))]
            dCdb[-1] = sigmaprime(z[-1]) * dCda[-1]

            # actual backprop
            for k in range(2, len(self.weights)):
                # first compute dCda[-k]
                dCda[-k] = 


def sigma(arr):
    return 1 / (1 + np.exp(-arr))

def sigmaprime(arr):
    np.exp(arr) / ((np.exp(arr) + 1) ** 2)

def evaluate

if __name__ == '__main__':
    start = time.time()
    g = gzip.open('trainingdata.gz', 'rb').read()
    numImages = sum((byte) * (256 ** i) for i, byte in enumerate(g[4:8][::-1]))
    imWidth = sum((byte) * (256 ** i) for i, byte in enumerate(g[8:12][::-1]))
    imHeight = sum((byte) * (256 ** i) for i, byte in enumerate(g[12:16][::-1]))
    g = g[16:]
    images = np.frombuffer(g, dtype=np.uint8).reshape(numImages, imHeight, imWidth)

    l = gzip.open('traininglabels.gz', 'rb').read()[8:]
    labels = np.array([int(el) for el in l])

    NUM_L1_NODES = imWidth * imHeight
    NUM_L2_NODES = 20
    NUM_L3_NODES = 20
    NUM_L4_NODES, NUM_DIGITS = 10, 10

    "Initialize weights and biases randomly"
    # weights[0] corresponds to link between L1 and L2, and so on
    weights = np.array([
        (-1 + 2*np.random.rand(NUM_L2_NODES, NUM_L1_NODES)),
        (-1 + 2*np.random.rand(NUM_L3_NODES, NUM_L2_NODES)),
        (-1 + 2*np.random.rand(NUM_L4_NODES, NUM_L3_NODES))
    ])
    # so weights[0][30][2] corresponds to the weight between node 30 in L2 and node 2 in L1

    biases = np.array([
        (-10 + 20*np.random.rand(NUM_L2_NODES)),
        (-10 + 20*np.random.rand(NUM_L3_NODES)),
        (-10 + 20*np.random.rand(NUM_L4_NODES))
    ])

    NN = Network(NUM_L1_NODES, NUM_L2_NODES, NUM_L3_NODES, NUM_L4_NODES, weights, biases)

    print('Completed in - {0} - seconds'.format(time.time() - start))
