#!/usr/bin/env python3

"""This module calculates the precision Confusion matrix."""

import numpy as np


def precision(confusion):
    """
    Calculates the precision (positive predictive value) for each class.

    Precision for a class is defined as: TP / (TP + FP)
    where TP = true positives for the class
          FP = false positives for the class

    In the confusion matrix:
    - True positives for class i are at position [i, i]
    - False positives for class i are the sum of column i excluding [i, i]

    Args:
        confusion: numpy.ndarray of shape (classes, classes) where
                   row indices represent correct labels and column indices
                   represent predicted labels

    Returns:
        numpy.ndarray of shape (classes,) containing the precision
        of each class. Returns 0 for classes with no predicted positive
        samples to avoid division by zero.

    Raises:
        ValueError: If confusion matrix is not square
    """
    # Check if confusion matrix is square
    if confusion.shape[0] != confusion.shape[1]:
        raise ValueError("Confusion matrix must be square")

    classes = confusion.shape[0]
    precisions = np.zeros(classes, dtype=np.float64)

    for i in range(classes):
        # True positives: diagonal element for class i
        tp = confusion[i, i]

        # False positives: sum of column i minus the diagonal element
        # (samples incorrectly predicted as class i)
        fp = np.sum(confusion[:, i]) - tp

        # Calculate precision, handling division by zero
        if tp + fp > 0:
            precisions[i] = tp / (tp + fp)
        else:
            # No samples predicted as this class
            precisions[i] = 0.0

    return precisions
