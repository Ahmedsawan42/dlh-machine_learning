#!/usr/bin/env python3


def cat_arrays(arr1, arr2):
    """
    Concatenate two arrays (lists) into a single list.

    Args:
        arr1 (list): The first array/list to be concatenated.
        arr2 (list): The second array/list to be concatenated.

    Returns:
        list: A new list containing all elements from arr1 followed by arr2.
    """
    result = []

    for element in arr1:
        result.append(element)

    for element in arr2:
        result.append(element)

    return result
