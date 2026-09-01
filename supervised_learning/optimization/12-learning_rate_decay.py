#!/usr/bin/env python3

"""This module create a learning rate decay operation in TensorFlow."""

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """
    Creates a learning rate decay operation in tf using inverse time decay.

    Args:
        alpha (float): Original learning rate.
        decay_rate (float): Weight, determine rate at which alpha will decay.
        decay_step (int): Number of passes of gradient descent should occur.
                         before alpha is decayed further.

    Returns:
        tensorflow.keras.optimizers.schedules.LearningRateSchedule: The
        learning rate decay operation.
    """
    # Create inverse time decay schedule with stepwise decay
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True
    )

    return lr_schedule
