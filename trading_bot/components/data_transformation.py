import pandas as pd
import numpy as np
import ta


class DataTransformation():

    def __init__(self):
        pass

    @staticmethod
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
    
    @staticmethod
    def create_candle_info_feats(df):
        '''Creates candle color, filling, and amplitude features

            Args:
                df (pd.DataFrame): original data

            Returns:
                pd.DataFrame: original data + 3 new columns/features
        '''
        df_copy = df.copy()

        # Candle color
        df_copy["candle_way"] = -1
        df_copy.loc[(df_copy["open"] - df_copy["close"]) < 0, "candle_way"] = 1

        # Filling percentage
        df_copy["filling"] = np.abs(df_copy["close"] - df_copy["open"]) / np.abs(df_copy["high"] - df_copy["low"])

        # Amplitude
        df_copy["amplitude"] = np.abs(df_copy["close"] - df_copy["open"]) / ((df_copy["open"] + df_copy["close"]) / 2) * 100

        return df_copy
    
    @staticmethod
    def create_kama_feat(df, col, n):
        '''Calculates KAMA indicator using ta library

           Args:
               df (pd.DataFrame): original data
               col (str): the column for which the indicator should be calculated

           Returns:
               pd.DataFrame: original data + a column with KAMA indicator
        '''
        df_copy = df.copy()
        df_copy[f"kama_{n}"] = ta.momentum.KAMAIndicator(df_copy[col], n).kama()
        return df_copy
    
    @staticmethod
    def derivatives(df,col):
        """
        Calculates the first and second derivatives of a given column in a DataFrame 
        and adds them as new columns 'velocity' and 'acceleration'.

        Args:
            df (pd.DataFrame): the DataFrame containing the column for which derivatives are to be calculated
            
            col (str): the column name for which the first and second derivatives are to be calculated

        Returns:
            pd.DataFrame: a new DataFrame with 'velocity' and 'acceleration' columns added

        """
        
        df_copy = df.copy()
        df_copy["velocity"] = df_copy[col].diff().fillna(0)
        df_copy["acceleration"] = df_copy["velocity"].diff().fillna(0)    
        return df_copy