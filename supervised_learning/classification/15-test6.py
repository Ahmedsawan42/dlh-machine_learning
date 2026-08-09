#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

NN = __import__('15-neural_network').NeuralNetwork

lib_train = np.load('../data/Binary_Train.npz')
X_train_3D, Y_train = lib_train['X'], lib_train['Y']
X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T

np.random.seed(1)
nn = NN(X_train.shape[0], 3)
nn.train(X_train, Y_train, iterations=1000)

# alexa@ubuntu:~$ ./test6.py 
# Cost after 0 iterations: 1.1471535092660408
# Cost after 100 iterations: 0.5774282684581176
# Cost after 200 iterations: 0.43800266354992273
# Cost after 300 iterations: 0.35239986589756944
# Cost after 400 iterations: 0.2911026286586329
# Cost after 500 iterations: 0.24523270781990958
# Cost after 600 iterations: 0.21032790651128522
# Cost after 700 iterations: 0.18300720260476488
# Cost after 800 iterations: 0.16111259651866097
# Cost after 900 iterations: 0.14331563506813214
# Cost after 1000 iterations: 0.12873028808003742
