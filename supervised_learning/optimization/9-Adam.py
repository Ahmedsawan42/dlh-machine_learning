#!/usr/bin/env python3

"""This module update variables using the Adam optimization algorithm."""

import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Updates a variable using the Adam optimization algorithm.

    Args:
        alpha (float): Learning rate.
        beta1 (float): Weight used for the first moment (0 < beta1 < 1).
        beta2 (float): Weight used for the second moment (0 < beta2 < 1).
        epsilon (float): Small number to avoid division by zero.
        var (numpy.ndarray): Variable to be updated.
        grad (numpy.ndarray): Gradient of var.
        v (numpy.ndarray): Previous first moment of var.
        s (numpy.ndarray): Previous second moment of var.
        t (int): Time step used for bias correction.

    Returns:
        tuple: (updated_var, new_v, new_s) containing the updated variable,
               the new first moment, and the new second moment, respectively.
    """
    # Update the first moment (momentum)
    v_new = beta1 * v + (1 - beta1) * grad

    # Update the second moment (RMSProp)
    s_new = beta2 * s + (1 - beta2) * (grad ** 2)

    # Apply bias correction to first moment
    v_corrected = v_new / (1 - beta1 ** t)

    # Apply bias correction to second moment
    s_corrected = s_new / (1 - beta2 ** t)

    # Update the variable
    var_new = var - alpha * (v_corrected / (np.sqrt(s_corrected) + epsilon))

    return var_new, v_new, s_new
