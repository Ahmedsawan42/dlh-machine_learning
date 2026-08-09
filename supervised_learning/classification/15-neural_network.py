#!/usr/bin/env python3
"""
Neural Network module of binary classification with one hidden layer
"""

import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    """
    Defines a neural network with one hidden layer of binary classification

    Attributes:
        nx (int): Number of input features
        nodes (int): Number of nodes in the hidden layer
    """

    def __init__(self, nx, nodes):
        """
        Class constructor of NeuralNetwork

        Args:
            nx (int): Number of input features
            nodes (int): Number of nodes in the hidden layer

        Raises:
            TypeError: If nx or nodes is not an integer
            ValueError: If nx or nodes is less than 1

        All exceptions are raised in the order listed above
        """
        # Check if nx is an integer
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")

        # Check if nx is positive
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Check if nodes is an integer
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")

        # Check if nodes is positive
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # Initialize private weights of hidden layer with random normal
        # Shape: (nodes, nx) - nodes neurons, each with nx weights
        self.__W1 = np.random.randn(nodes, nx)

        # Initialize private bias of hidden layer with zeros
        # Shape: (nodes, 1) - one bias per neuron in hidden layer
        self.__b1 = np.zeros((nodes, 1))

        # Initialize private activated output of hidden layer to 0
        self.__A1 = 0

        # Initialize private weights of output neuron with random normal
        # Shape: (1, nodes) - 1 output neuron, with nodes weights
        self.__W2 = np.random.randn(1, nodes)

        # Initialize private bias of output neuron to 0
        self.__b2 = 0

        # Initialize private activated output of output neuron to 0
        self.__A2 = 0

    @property
    def W1(self):
        """
        Getter of private attribute __W1

        Returns:
            numpy.ndarray: The weights matrix of the hidden layer
        """
        return self.__W1

    @property
    def b1(self):
        """
        Getter of private attribute __b1

        Returns:
            numpy.ndarray: The bias vector of the hidden layer
        """
        return self.__b1

    @property
    def A1(self):
        """
        Getter of private attribute __A1

        Returns:
            numpy.ndarray: The activated output of the hidden layer
        """
        return self.__A1

    @property
    def W2(self):
        """
        Getter of private attribute __W2

        Returns:
            numpy.ndarray: The weights vector of the output neuron
        """
        return self.__W2

    @property
    def b2(self):
        """
        Getter of private attribute __b2

        Returns:
            float: The bias of the output neuron
        """
        return self.__b2

    @property
    def A2(self):
        """
        Getter of private attribute __A2

        Returns:
            numpy.ndarray: The activated output of the output neuron
        """
        return self.__A2

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network

        Args:
            X (numpy.ndarray): Input data with shape (nx, m)
                               nx = number of input features
                               m = number of examples

        Returns:
            tuple: (__A1, __A2) - activated outputs of hidden layer and
                   output layer respectively

        Updates:
            Private attributes __A1 and __A2
        """
        # Hidden layer forward propagation
        # Z1 = W1·X + b1
        # W1 shape: (nodes, nx), X shape: (nx, m)
        # Result shape: (nodes, m)
        Z1 = np.dot(self.__W1, X) + self.__b1

        # Apply sigmoid activation of hidden layer
        # A1 = σ(Z1)
        self.__A1 = 1 / (1 + np.exp(-Z1))

        # Output layer forward propagation
        # Z2 = W2·A1 + b2
        # W2 shape: (1, nodes), A1 shape: (nodes, m)
        # Result shape: (1, m)
        Z2 = np.dot(self.__W2, self.__A1) + self.__b2

        # Apply sigmoid activation of output layer
        # A2 = σ(Z2)
        self.__A2 = 1 / (1 + np.exp(-Z2))

        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression

        Args:
            Y (numpy.ndarray): Correct labels with shape (1, m)
            A (numpy.ndarray): Activated output with shape (1, m)

        Returns:
            float: The cross-entropy cost

        Note:
            Uses 1.0000001 - A instead of 1 - A to avoid division by zero
        """
        # Number of examples
        m = Y.shape[1]

        # Calculate the cost using cross-entropy loss
        # Cost = -1/m * sum(Y * log(A) + (1-Y) * log(1-A))
        # Use 1.0000001 - A to avoid division by zero errors
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))

        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network's predictions

        Args:
            X (numpy.ndarray): Input data with shape (nx, m)
                               nx = number of input features
                               m = number of examples
            Y (numpy.ndarray): Correct labels with shape (1, m)

        Returns:
            tuple: (prediction, cost)
                prediction: numpy.ndarray with shape (1, m) containing
                           predicted labels (0 or 1)
                cost: float representing the cross-entropy cost
        """
        # Perform forward propagation to get activated outputs
        A1, A2 = self.forward_prop(X)

        # Convert probabilities to binary predictions
        # 1 if A2 >= 0.5, 0 otherwise
        prediction = np.where(A2 >= 0.5, 1, 0)

        # Calculate the cost using the output layer activations
        cost = self.cost(Y, A2)

        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network

        Args:
            X (numpy.ndarray): Input data with shape (nx, m)
            Y (numpy.ndarray): Correct labels with shape (1, m)
            A1 (numpy.ndarray): Output the hidden layer with shape (nodes, m)
            A2 (numpy.ndarray): Predicted output with shape (1, m)
            alpha (float): Learning rate (default 0.05)

        Updates:
            Private attributes __W1, __b1, __W2, and __b2
        """
        # Number of examples
        m = X.shape[1]

        # Output layer gradients
        # dZ2 = A2 - Y
        dZ2 = A2 - Y

        # dW2 = (1/m) * dZ2 · A1^T
        # dZ2 shape: (1, m), A1^T shape: (m, nodes)
        # Result shape: (1, nodes)
        dW2 = (1 / m) * np.dot(dZ2, A1.T)

        # db2 = (1/m) * sum(dZ2)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

        # Hidden layer gradients
        # dZ1 = W2^T · dZ2 * σ'(Z1)
        # σ'(Z1) = A1 * (1 - A1)
        # W2^T shape: (nodes, 1), dZ2 shape: (1, m)
        # Result shape: (nodes, m)
        dZ1 = np.dot(self.__W2.T, dZ2) * (A1 * (1 - A1))

        # dW1 = (1/m) * dZ1 · X^T
        # dZ1 shape: (nodes, m), X^T shape: (m, nx)
        # Result shape: (nodes, nx)
        dW1 = (1 / m) * np.dot(dZ1, X.T)

        # db1 = (1/m) * sum(dZ1)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        # Update weights and biases using gradient descent
        # W1 = W1 - alpha * dW1
        self.__W1 = self.__W1 - alpha * dW1

        # b1 = b1 - alpha * db1
        self.__b1 = self.__b1 - alpha * db1

        # W2 = W2 - alpha * dW2
        self.__W2 = self.__W2 - alpha * dW2

        # b2 = b2 - alpha * db2
        self.__b2 = self.__b2 - alpha * db2

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """
        Trains the neural network

        Args:
            X (numpy.ndarray): Input data with shape (nx, m)
            Y (numpy.ndarray): Correct labels with shape (1, m)
            iterations (int): Number of iterations to train over(default 5000)
            alpha (float): Learning rate (default 0.05)
            verbose (bool): Whether print training information (default True)
            graph (bool): Whether to plot training cost (default True)
            step (int): Steps between verbose output and graph points
                        (default 100)

        Returns:
            tuple: (prediction, cost) - evaluation of training data after
                   iterations of training

        Raises:
            TypeError: If iterations is not an integer, alpha is not a float,
                      or step is not an integer (when verbose or graph is True)
            ValueError: If iterations is not positive, alpha is not positive,
                       or step is not positive and <= iterations
                       (when verbose or graph is True)

        Updates:
            Private attributes __W1, __b1, __A1, __W2, __b2, and __A2
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

        # Check step parameter only if verbose or graph is True
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        # Initialize lists to store costs and iterations for plotting
        costs = []
        iter_list = []

        # Get initial cost (iteration 0)
        A1, A2 = self.forward_prop(X)
        initial_cost = self.cost(Y, A2)

        if verbose:
            print(f"Cost after 0 iterations: {initial_cost}")

        if graph:
            costs.append(initial_cost)
            iter_list.append(0)

        # Training loop
        for i in range(1, iterations + 1):
            # Forward propagation
            A1, A2 = self.forward_prop(X)

            # Gradient descent
            self.gradient_descent(X, Y, A1, A2, alpha)

            # Recalculate forward propagation with updated weights
            _, current_A2 = self.forward_prop(X)

            # Check if we need to record cost at this iteration
            if (verbose or graph) and i % step == 0:
                # Calculate current cost using updated activation
                current_cost = self.cost(Y, current_A2)

                if verbose:
                    print(f"Cost after {i} iterations: {current_cost}")

                if graph:
                    costs.append(current_cost)
                    iter_list.append(i)

        # Get final cost (last iteration) if not already recorded
        if graph and iterations % step != 0:
            final_cost = self.cost(Y, self.__A2)
            costs.append(final_cost)
            iter_list.append(iterations)

        # Plot graph if requested
        if graph:
            plt.plot(iter_list, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.savefig("15-neural_network.png")
            plt.show()

        # Evaluate the training data after training
        return self.evaluate(X, Y)
