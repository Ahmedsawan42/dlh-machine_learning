#!/usr/bin/env python3

"""This module update variables using the RMSProp optimization algorithm."""

import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Updates a variable using the RMSProp optimization algorithm.

    Args:
        alpha (float): Learning rate.
        beta2 (float): RMSProp weight (0 < beta2 < 1).
        epsilon (float): Small number to avoid division by zero.
        var (numpy.ndarray): Variable to be updated.
        grad (numpy.ndarray): Gradient of var.
        s (numpy.ndarray): Previous second moment of var.

    Returns:
        tuple: (updated_var, new_s) containing the updated variable and
               the new moment, respectively.
    """
    # Update the exponentially weighted average of squared gradients
    # (second moment)
    s_new = beta2 * s + (1 - beta2) * (grad ** 2)

    # Update the variable using RMSProp
    var_new = var - alpha * (grad / (np.sqrt(s_new) + epsilon))

    return var_new, s_new
