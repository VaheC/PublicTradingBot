import pandas as pd
import numpy as np
import ta
import math


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
    def create_derivatives_feats(df,col):
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
    
    @staticmethod
    def create_spread_feat(df):
        """
        Calculates the spread between the 'high' and 'low' columns of a given DataFrame 
        and adds it as a new column named 'spread'.

        Args:
            df (pd.DataFrame): the DataFrame containing the 'high' and 'low' columns for which the spread is to be calculated

        Returns:
            pd.DataFrame: a new DataFrame with the 'spread' column added
        """
        df_copy = df.copy()
        df_copy["spread"] = df_copy["high"] - df_copy["low"]
        return df_copy
    
    @staticmethod
    def create_moving_parkinson_estimator_feat(df, window_size=30):
        """
        Calculate Parkinson's volatility estimator based on high and low prices.

        Args:
            df (pd.DataFrame): DataFrame containing 'high' and 'low' columns for each trading period
            window_size (int): size of moving window for the calculation

        Returns:
            pd.DataFrame: original data + a column with the estimated volatility based on Parkinson's method
        """
        def parkinson_estimator(df):
            N = len(df)
            sum_squared = np.sum(np.log(df['high'] / df['low']) ** 2)

            volatility = math.sqrt((1 / (4 * N * math.log(2))) * sum_squared)
            return volatility
        
        df_copy = df.copy()
        # Create an empty series to store mobile volatility
        rolling_volatility = pd.Series(dtype='float64')

        # Browse the DataFrame by window size `window_size` and apply `parkinson_estimator`.
        for i in range(window_size, len(df)):
            window = df_copy.loc[df_copy.index[i-window_size]: df_copy.index[i]]
            volatility = parkinson_estimator(window)
            rolling_volatility.at[df_copy.index[i]] = volatility

        # Add the mobile volatility series to the original DataFrame
        df_copy['rolling_volatility_parkinson'] = rolling_volatility
        
        return df_copy
    
    @staticmethod
    def create_moving_yang_zhang_estimator_feat(df, window_size=30):
        """
        Calculate Yang-Zhang's volatility estimator based on high and low prices.

        Args:
            df (pd.DataFrame): DataFrame containing 'high' and 'low' columns for each trading period
            window_size (int): size of moving window for the calculation

        Returns:
            pd.DataFrame: original data + a column with the estimated volatility based on Yang-Zhang's method
        """
        def yang_zhang_estimator(df):
            N = len(window)
        
            term1 = np.log(window['high'] / window['close']) * np.log(window['high'] / window['open'])
            term2 = np.log(window['low'] / window['close']) * np.log(window['low'] / window['open'])

            sum_squared = np.sum(term1 + term2)
            volatility = np.sqrt(sum_squared / N)

            return volatility
        
        df_copy = df.copy()
        
        # Create an empty series to store mobile volatility
        rolling_volatility = pd.Series(dtype='float64')

        # Browse the DataFrame by window size `window_size` and apply `yang_zhang_estimator`.
        for i in range(window_size, len(df)):
            window = df_copy.loc[df_copy.index[i-window_size]: df_copy.index[i]]
            volatility = yang_zhang_estimator(window)
            rolling_volatility.at[df_copy.index[i]] = volatility

        # Add the mobile volatility series to the original DataFrame
        df_copy['rolling_volatility_yang_zhang'] = rolling_volatility
        
        return df_copy
    
    @staticmethod
    def create_displacement_detection_feats(df, type_range="standard", strengh=3, period=100):
        """
        This function calculates and adds a 'displacement' column to a provided DataFrame. Displacement is determined based on
        the 'candle_range' which is calculated differently according to the 'type_range' parameter. Then, it calculates the
        standard deviation of the 'candle_range' over a given period and sets a 'threshold'. If 'candle_range' exceeds this 
        'threshold', a displacement is detected and marked as 1 in the 'displacement' column.

        Args:
            df (pd.DataFrame): the DataFrame to add the columns to, this DataFrame should have 'open', 'close', 'high', 
                               and 'low' columns
            type_range (str, optional): defines how to calculate 'candle_range'. 'standard' calculates it as the absolute difference 
                                        between 'close' and 'open', 'extremum' calculates it as the absolute difference between 
                                        'high' and 'low', default is 'standard'
            strengh (int, optional): the multiplier for the standard deviation to set the 'threshold', default is 3
            period (int, optional): the period to use for calculating the standard deviation, default is 100

        Returns:
            pd.DataFrame: the original DataFrame, but with four new columns: 'candle_range', 'MSTD', 'threshold' and 'displacement'

        Raises:
        ValueError: If an unsupported 'type_range' is provided.
        """
        df_copy = df.copy()

        # Choose your type_range
        if type_range == "standard":
            df_copy["candle_range"] = np.abs(df_copy["close"] - df_copy["open"])
        elif type_range == "extremum":
            df_copy["candle_range"] = np.abs(df_copy["high"] - df_copy["low"])
        else:
            raise ValueError("Put a right format of type range")

        # Compute the STD of the candle range
        df_copy["MSTD"] = df_copy["candle_range"].rolling(period).std()
        df_copy["threshold"] = df_copy["MSTD"] * strengh

        # Displacement if the candle range is above the threshold
        df_copy["displacement"] = np.nan
        df_copy.loc[df_copy["threshold"] < df_copy["candle_range"], "displacement"] = 1
        df_copy["variation"] = df_copy["close"] - df_copy["open"]

        # Specify the way of the displacement
        df_copy["green_displacement"] = 0
        df_copy["red_displacement"] = 0

        df_copy.loc[(df_copy["displacement"] == 1) & (0 < df_copy["variation"]), "green_displacement"] = 1
        df_copy.loc[(df_copy["displacement"] == 1) & (df_copy["variation"] < 0), "red_displacement"] = 1

        # Shift by one because we only know that we have a displacement at the end of the candle (BE CAREFUL)
        df_copy["green_displacement"] = df_copy["green_displacement"].shift(1)
        df_copy["red_displacement"] = df_copy["red_displacement"].shift(1)

        df_copy["high_displacement"] = np.nan
        df_copy["low_displacement"] = np.nan

        df_copy.loc[df_copy["displacement"] == 1, "high_displacement"] = df_copy["high"]
        df_copy.loc[df_copy["displacement"] == 1, "low_displacement"] = df_copy["low"]

        df_copy["high_displacement"] = df_copy["high_displacement"].fillna(method="ffill")
        df_copy["low_displacement"] = df_copy["low_displacement"].fillna(method="ffill")

        return df_copy