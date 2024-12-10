from src.stockpredictor.utils.common import save_as_csv
from src.stockpredictor.logging.coustom_log import logger
from src.stockpredictor.entity import ModelTrainingConfig
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

class ModelTrainer:
    def __init__(self, config: ModelTrainingConfig) -> None:
        self.config = config

    def build_and_complile_lstm(self, input_shape):
        model = Sequential()
        model.add(LSTM(units = 50, activation = "relu", return_sequences = True, input_shape = input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(units = 60, activation = "relu", return_sequences = True))
        model.add(Dropout(0.2))
        model.add(LSTM(units = 80, activation = "relu", return_sequences = True))
        model.add(Dropout(0.4))
        model.add(LSTM(units= 120, activation = "relu"))
        model.add(Dropout(0.5))

        model.compile(optimizer = "adam", loss = "mean_squared_error")

        return model

    def train_LSTM_model(self):
        # Load the transformed data
        stock_data = pd.read_csv(self.config.training_input)

        # Split into training and test sets
        train_data = pd.DataFrame(stock_data.Close[0: int(len(stock_data)*0.80)])
        test_data = pd.DataFrame(stock_data.Close[int(len(stock_data)*0.80): len(stock_data)])

        # Feature scaling
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_train_data = scaler.fit_transform(train_data)

        # Prepare data for LSTM
        x = []
        y = []

        # Using 100 previous days to predict the next day
        for i in range(100, scaled_train_data.shape[0]):
            x.append(scaled_train_data[i-100:i])  # 100 previous days as features
            y.append(scaled_train_data[i, 0])     # Next day's closing price as target

        x, y = np.array(x), np.array(y)

        # Reshape x to be 3D for LSTM input: [samples, time steps, features]
        x = np.reshape(x, (x.shape[0], x.shape[1], 1))

        # Build and compile LSTM model
        lstm_model = self.build_and_complile_lstm(input_shape=(x.shape[1], x.shape[2]))

        # Reshape y to be 2D for the output: [samples, 1]
        y = np.reshape(y, (y.shape[0], 1))

        # Train the model
        lstm_model.fit(x, y, epochs=self.config.lstm_epochs, batch_size=self.config.lstm_batch_size, verbose=self.config.lstm_verbose)
        lstm_model.save(self.config.lstm_model)
        

        



