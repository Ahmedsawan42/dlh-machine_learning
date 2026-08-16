#!/usr/bin/env python3

"""Module for training a Keras model using mini-batch gradient descent"""

import tensorflow.keras as K


def train_model(
    network,
    data,
    labels,
    batch_size,
    epochs,
    verbose=True,
    shuffle=False
):
    """
    Trains a model using mini-batch gradient descent

    Args:
        network: the model to train
        data: np.ndarray of shape (m, nx) containing the input data
        labels: one-hot np.ndarray of shape (m, classes) containing the labels
        batch_size: size of the batch used for mini-batch gradient descent
        epochs: number of passes through data for mini-batch gradient descent
        verbose: boolean determines if output should printed during training
        shuffle: boolean determines whether to shuffle the batches every epoch

    Returns:
        The History object generated after training the model
    """
    # Train the model using the fit method
    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle
    )

    return history
