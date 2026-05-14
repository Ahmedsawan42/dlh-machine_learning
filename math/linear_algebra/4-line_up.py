#!/usr/bin/env python3


def add_arrays(arr1, arr2):
    """
    This function takes two lists of equal length and Performelement-wise
    addition of two arrays (lists) of numbers.

    Args:
        arr1 (list of numeric): The first array/list to be added.
        arr2 (list of numeric): The second array/list to be added.

    Returns:
        list of int: A new list containing the element-wise sums.
    """
    if len(arr1) != len(arr2):
        return None

    result = []

    for i in range(len(arr1)):
        sum_of_elements = arr1[i] + arr2[i]
        result.append(sum_of_elements)

    return result
