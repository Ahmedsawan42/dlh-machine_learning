#!/usr/bin/env python3

"""a Model to loads data from a file as a pd.DataFrame"""

import pandas as pd


def from_file(filename, delimiter):
    """
    loads data from a file as a pd.DataFrame.

    Args:
        filename: the file to load from
        delimiter: the column separator

    Returns: the loaded pd.DataFrame
    """
    df = pd.read_csv(filename, delimiter=delimiter)
    return df
