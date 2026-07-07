#!/usr/bin/env python3

"""a Model that creates a np.ndarray from a pd.DataFrame"""


def array(df):
    """
    Selects the last 10 rows of the High and Close columns
    and converts them to a numpy.ndarray.

    Args:
        df: a pd.DataFrame containing columns named High and Close

    Returns: the numpy.ndarray
    """
    # Select the last 10 rows of High and Close columns
    selected_data = df[['High', 'Close']].tail(10)

    # Convert to numpy array
    result = selected_data.to_numpy()
    # result = np.array(selected_data)

    return result
