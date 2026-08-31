#!/usr/bin/env python3

"""This module update variables using gradient descent with momentum."""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Updates a variable using the gradient descent with momentum optimization.

    Args:
        alpha (float): Learning rate.
        beta1 (float): Momentum weight (0 < beta1 < 1).
        var (numpy.ndarray): Variable to be updated.
        grad (numpy.ndarray): Gradient of var.
        v (numpy.ndarray): Previous first moment of var.

    Returns:
        tuple: (updated_var, new_v) containing the updated variable and
               the new moment, respectively.
    """
    # Update the momentum (first moment)
    v_new = beta1 * v + (1 - beta1) * grad

    # Update the variable
    var_new = var - alpha * v_new

    return var_new, v_new
