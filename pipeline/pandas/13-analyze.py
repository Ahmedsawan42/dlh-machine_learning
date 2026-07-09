#!/usr/bin/env python3

"""a Model that computes descriptive statistics for all columns"""


def analyze(df):
    """
    Computes descriptive statistics for all columns except Timestamp column.

    Args: df: a pd.DataFrame containing a column named Timestamp

    Returns: a new pd.DataFrame containing these statistics
    """
    # Select numeric columns and exclude Timestamp
    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    # Remove Timestamp if it's somehow numeric (unlikely but just in case)
    if 'Timestamp' in numeric_cols:
        numeric_cols.remove('Timestamp')

    # Compute descriptive statistics for numeric columns
    stats_df = df[numeric_cols].describe()

    return stats_df
