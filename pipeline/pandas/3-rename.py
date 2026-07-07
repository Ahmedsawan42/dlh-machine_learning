#!/usr/bin/env python3

"""a Model to Renames Timestamp column to Datetime"""

import pandas as pd


def rename(df):
    """
    Renames Timestamp column to Datetime, converts to datetime values,
    and displays only Datetime and Close columns.

    Args: df: a pd.DataFrame containing a column named Timestamp

    Returns: the modified pd.DataFrame with only Datetime and Close columns
    """
    # Rename the column from 'Timestamp' to 'Datetime'
    df = df.rename(columns={'Timestamp': 'Datetime'})

    # Convert the Datetime column to datetime values
    # The timestamps are in seconds since the Unix epoch
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')

    # Keep only Datetime and Close columns
    df = df[['Datetime', 'Close']]

    return df
