#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

Neuron = __import__('7-neuron').Neuron

lib_train = np.load('../data/Binary_Train.npz')
X_train_3D, Y_train = lib_train['X'], lib_train['Y']
X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T

np.random.seed(1)
neuron = Neuron(X_train.shape[0])
neuron.train(X_train, Y_train, iterations=105, graph=False)

# alexa@ubuntu:~$ ./test2.py 
# Cost after 0 iterations: 3.898941233712408
# Cost after 100 iterations: 0.22327183152859484
# Cost after 105 iterations: 0.21110792491802174
