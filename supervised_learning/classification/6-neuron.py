#!/usr/bin/env python3

"""a Model that defines a single neuron performing binary classification"""

import numpy as np


class Neuron:
    def __init__(self, nx):
        # Check if nx is an integer
        if not isinstance(nx, int):
            raise TypeError("nx must be a integer")

        # Check if nx is positive
        if nx < 1:
            raise ValueError("nx must be positive")

        # Initialize private weights with random normal distribution (2D array)
        self.__W = np.random.randn(1, nx)

        # Initialize private bias to 0
        self.__b = 0

        # Initialize private activated output to 0
        self.__A = 0

    # Getter for W
    @property
    def W(self):
        return self.__W

    # Getter for b
    @property
    def b(self):
        return self.__b

    # Getter for A
    @property
    def A(self):
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron

        Args:
            X: numpy.ndarray with shape (nx, m) containing the input data
               nx = number of input features
               m = number of examples

        Returns:
            The private attribute __A (activated output)
        """
        # Calculate the weighted sum: z = W·X + b
        z = np.dot(self.__W, X) + self.__b

        # Apply sigmoid activation function
        self.__A = 1 / (1 + np.exp(-z))

        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression

        Args:
            Y: numpy.ndarray with shape (1, m) containing the correct labels
            A: numpy.ndarray with shape (1, m) containing the activated output

        Returns:
            The cost (cross-entropy loss)
        """
        # Number of examples
        m = Y.shape[1]

        # Calculate the cost using cross-entropy loss
        # Cost = -1/m * sum(Y * log(A) + (1-Y) * log(1-A))
        # Use 1.0000001 - A to avoid division by zero errors
        cost = -1/m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))

        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neuron's predictions

        Args:
            X: numpy.ndarray with shape (nx, m) containing the input data
            Y: numpy.ndarray with shape (1, m) containing the correct labels

        Returns:
            prediction: array with shape (1, m) containing predicted labels
            cost: the cost of the network
        """
        # Perform forward propagation to get activated outputs
        A = self.forward_prop(X)

        # Convert probabilities to binary predictions
        # 1 if A >= 0.5, 0 otherwise
        prediction = np.where(A >= 0.5, 1, 0)

        # Calculate the cost using the activated outputs
        cost = self.cost(Y, A)

        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron

        Args:
            X: numpy.ndarray with shape (nx, m) containing the input data
            Y: numpy.ndarray with shape (1, m) containing the correct labels
            A: numpy.ndarray with shape (1, m) containing the activated output
            alpha: learning rate (default 0.05)

        Updates:
            Private attributes __W and __b
        """
        # Number of examples
        m = X.shape[1]

        # Calculate the gradient of the weights
        # dW = (1/m) * (A - Y) · X^T
        # (A - Y) has shape (1, m), X^T has shape (m, nx)
        # Result has shape (1, nx)
        dW = (1/m) * np.dot((A - Y), X.T)

        # Calculate the gradient of the bias
        # db = (1/m) * sum(A - Y)
        db = (1/m) * np.sum(A - Y)

        # Update weights and bias using gradient descent
        # W = W - alpha * dW
        # b = b - alpha * db
        self.__W = self.__W - alpha * dW
        self.__b = self.__b - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the neuron

        Args:
            X: numpy.ndarray with shape (nx, m) containing the input data
            Y: numpy.ndarray with shape (1, m) containing the correct labels
            iterations: number of iterations to train over (default 5000)
            alpha: learning rate (default 0.05)

        Returns:
            The evaluation of the training data after iterations of training
        """
        # Check if iterations is an integer
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")

        # Check if iterations is positive
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")

        # Check if alpha is a float
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")

        # Check if alpha is positive
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        # Training loop
        for _ in range(iterations):
            # Forward propagation
            A = self.forward_prop(X)

            # Gradient descent
            self.gradient_descent(X, Y, A, alpha)

        # Evaluate the training data after training
        return self.evaluate(X, Y)
