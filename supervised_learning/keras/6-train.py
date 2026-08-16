#!/usr/bin/env python3

"""Module for training a Keras model with early stopping and validation"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent with optional
    validation and early stopping

    Args:
        network: the model to train
        data: numpy.ndarray of shape (m, nx) containing the input data
        labels: one-hot numpy.ndarray of shape (m, classes) containing
                the labels
        batch_size: size of the batch used for mini-batch gradient descent
        epochs: number of passes through data for mini-batch gradient descent
        validation_data: data to validate the model with, if not None
        early_stopping: boolean indicating if early stopping should be used
        patience: patience used for early stopping
        verbose: boolean that determines if output should be printed
                 during training
        shuffle: boolean that determines whether to shuffle the batches
                 every epoch

    Returns:
        The History object generated after training the model
    """
    # Prepare validation data if provided
    validation_split = 0.0
    validation_data_tuple = None

    if validation_data is not None:
        # If validation_data is a tuple (x_val, y_val), use it directly
        if isinstance(validation_data, tuple) and len(validation_data) == 2:
            validation_data_tuple = validation_data
        else:
            # If validation_data is a single value, use it as validation split
            validation_split = validation_data

    # Set up early stopping callback if conditions are met
    callbacks = []
    if early_stopping and validation_data is not None:
        early_stopping_callback = K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience
        )
        callbacks.append(early_stopping_callback)

    # Train the model using the fit method
    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
        validation_data=validation_data_tuple,
        validation_split=validation_split,
        callbacks=callbacks
    )

    return history
