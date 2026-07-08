#!/usr/bin/env python3

"""a Model that removes any entries where Close has NaN values"""


def prune(df):
    """
    Removes any entries where Close has NaN values.

    Args: df: a pd.DataFrame containing a column named Close

    Returns: the modified pd.DataFrame
    """
    return df.dropna(subset=['Close'])
