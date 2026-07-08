#!/usr/bin/env python3

"""a Model that sets the Timestamp column as the index of the dataframe"""


def index(df):
    """
    Sets the Timestamp column as the index of the dataframe.

    Args: df: a pd.DataFrame containing a column named Timestamp

    Returns: the modified pd.DataFrame
    """
    return df.set_index('Timestamp')
