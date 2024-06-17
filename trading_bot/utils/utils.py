import pandas as pd


def resample_data(df, conversion_tframe):
    '''Converts data to the specified frequency

       Args:
           df (dataframe): contains the original data
           conversion_tframe (str): specifies the conversion timeframe (15T, 30T, 1H, 4H, 1D, 1W, 1M, etc.)

        Returns:
           dataframe: resampled dataframe
    '''
    resample = df.resample(conversion_tframe)

    df_resampled = pd.DataFrame()
    df_resampled["open"] = resample["open"].first()
    df_resampled["high"] = resample["high"].max()
    df_resampled["low"] = resample["low"].min()
    df_resampled["close"] = resample["close"].last()
    df_resampled["volume"] = resample["volume"].sum()

    return df_resampled