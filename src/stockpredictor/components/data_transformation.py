import os
from pathlib import Path
import yfinance as yf
from src.stockpredictor.utils.common import save_as_csv
from src.stockpredictor.logging.coustom_log import logger
from src.stockpredictor.entity import DataTransformationConfig
import pandas as pd
import pandas as pd

class DataTransformation():
    def __init__(self, config: DataTransformationConfig) -> None:
        self.config = config

    def calculate_rsi(self, df, window=14):
        try:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception as e:
            raise e

    def add_features(self, validated_dataframe: pd.DataFrame) -> pd.DataFrame:
        try:
            # Transform the values to numeric
            # Convert specific columns to numeric
            # columns_to_convert = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
            # for col in columns_to_convert:
            #     validated_dataframe[col] = pd.to_numeric(validated_dataframe[col], errors='coerce')


            validated_dataframe['Daily Returns'] = validated_dataframe['Close'].pct_change()
            validated_dataframe['High/Low Ratio'] = validated_dataframe['High'] / validated_dataframe['Low']
            validated_dataframe['Close/Open Ratio'] = validated_dataframe['Close'] / validated_dataframe['Open']
            validated_dataframe['Cumulative Returns'] = (1 + validated_dataframe['Daily Returns']).cumprod() - 1

            # Lag values for Close
            validated_dataframe['lag_1'] = validated_dataframe['Close'].shift(1)
            validated_dataframe['lag_2'] = validated_dataframe['Close'].shift(2)
            validated_dataframe['lag_7'] = validated_dataframe['Close'].shift(7)
            validated_dataframe['lag_30'] = validated_dataframe['Close'].shift(30)

            # Rolling statistical features
            validated_dataframe['moving_avg_5'] = validated_dataframe['Close'].rolling(window=5).mean()
            validated_dataframe['moving_avg_10'] = validated_dataframe['Close'].rolling(window=10).mean()
            validated_dataframe['moving_avg_30'] = validated_dataframe['Close'].rolling(window=30).mean()

            validated_dataframe['volatility_5'] = validated_dataframe['Close'].rolling(window=5).std()
            validated_dataframe['volatility_10'] = validated_dataframe['Close'].rolling(window=10).std()

            # Relative Strength Index
            validated_dataframe['rsi'] = self.calculate_rsi(validated_dataframe)

            # Add market sentiment and trend indicators
            validated_dataframe['price_change_pct'] = validated_dataframe['Close'].pct_change() * 100
            validated_dataframe['day_of_week'] = validated_dataframe['Date'].dt.dayofweek
            validated_dataframe['month'] = validated_dataframe['Date'].dt.month

            # Price volatality Indicators(True range and average true range)
            validated_dataframe['true_range'] = validated_dataframe['High'] - validated_dataframe['Low']
            validated_dataframe['atr'] = validated_dataframe['true_range'].rolling(window=14).mean() 

            # remove any none values created by features like rooling statistics
            transformaed_dataframe = validated_dataframe.dropna()
            return transformaed_dataframe
        except Exception as e:
            raise e
    
    def transform_data(self):
        try:
            validated_dataframe = pd.read_csv(self.config.transformation_input, parse_dates=['Date'])
            transformed_dataframe = self.add_features(validated_dataframe)
            save_as_csv(transformed_dataframe, self.config.transformation_output)
        except Exception as e:
            raise e