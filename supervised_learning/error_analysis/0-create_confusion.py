#!/usr/bin/env python3

"""This module creates a confusion matrix from one-hot encoded labels"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix from one-hot encoded labels and predictions.

    Args:
        labels: np.ndarray of shape (m, classes) containing correct labels
        logits: np.ndarray of shape (m, classes) containing predicted labels

    Returns:
        numpy.ndarray of shape (classes, classes) where:
        - Row indices represent the correct labels (true labels)
        - Column indices represent the predicted labels
    """
    # Get the number of samples and classes
    m, classes = labels.shape

    # Convert one-hot to class indices
    true_labels = np.argmax(labels, axis=1)
    pred_labels = np.argmax(logits, axis=1)

    # Initialize confusion matrix with zeros
    confusion_matrix = np.zeros((classes, classes), dtype=np.float64)

    # Fill the confusion matrix
    for true, pred in zip(true_labels, pred_labels):
        confusion_matrix[true, pred] += 1

    return confusion_matrix
