import pandas as pd


def create_auto_corr_feat(df, col, n=50, lag=10):
    """
    Calculates the autocorrelation for a given column in a Pandas DataFrame, using a specified window size and lag.

    Args:
        df (pd.DataFrame): Input DataFrame containing the column for which to compute autocorrelation.
        col (str): The name of the column in the DataFrame for which to calculate autocorrelation.
        n (int, optional): The size of the rolling window for calculation. Default is 50.
        lag (int, optional): The lag step to be used when computing autocorrelation. Default is 10.

    Returns:
        pd.DataFrame: A new DataFrame with an additional column named 'autocorr_{lag}', where {lag} is the provided lag value. 
                      This column contains the autocorrelation values.
    """
    df_copy = df.copy()
    df_copy[f'autocorr_{lag}'] = df_copy[col].rolling(window=n, min_periods=n, center=False).\
        apply(lambda x: x.autocorr(lag=lag), raw=False)
    return df_copy