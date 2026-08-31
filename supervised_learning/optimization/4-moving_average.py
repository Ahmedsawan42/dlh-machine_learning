#!/usr/bin/env python3

"""This module calculate weighted moving average with bias correction."""


def moving_average(data, beta):
    """
    Calculates the weighted moving average of a data set with bias correction.

    Args:
        data (list): List of data to calculate the moving average of.
        beta (float): Weight used for the moving average (0 < beta < 1).

    Returns:
        list: list containing the moving averages of data with bias correction
    """
    # Initialize the moving average variable
    v = 0

    # Initialize list to store moving averages
    moving_avg = []

    # Iterate through each data point
    for t, value in enumerate(data, 1):
        # Update the exponentially weighted average
        v = beta * v + (1 - beta) * value

        # Apply bias correction
        v_corrected = v / (1 - beta ** t)

        # Append the bias-corrected value to the list
        moving_avg.append(v_corrected)

    return moving_avg
