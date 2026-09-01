#!/usr/bin/env python3

"""This module update learning rate using inverse time decay. """

import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Updates the learning rate using inverse time decay in numpy.

    Args:
        alpha (float): Original learning rate.
        decay_rate (float): Weight, determine rate at which alpha will decay.
        global_step (int): Number of passes that have elapsed.
        decay_step (int): Number of passes of gradient descent should occur.
                         before alpha is decayed further.

    Returns:
        float: The updated value for alpha.
    """
    # Calculate the number of decay steps that have occurred
    # Using floor division to achieve stepwise decay
    decay_factor = np.floor(global_step / decay_step)

    # Apply inverse time decay
    updated_alpha = alpha / (1 + decay_rate * decay_factor)

    return updated_alpha
