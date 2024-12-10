from src.stockpredictor.utils.common import save_as_csv
from src.stockpredictor.logging.coustom_log import logger
from src.stockpredictor.entity import ModelTrainingConfig
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from keras.callbacks import EarlyStopping


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
        model.add(Dense(units = 10))

        model.compile(optimizer = "adam", loss = "mean_squared_error")

        return model

    def train_LSTM_model(self):
        # Load the transformed data
        stock_data = pd.read_csv(self.config.training_input)
        
        features = ['Close', 'moving_avg_10', 'moving_avg_30', 'rsi', 'volatility_5', 'volatility_10', 'atr', 'price_change_pct', 'day_of_week', 'month']
        stock_data = stock_data[features]

        # Split into training and test sets
        train_data = stock_data[0: int(len(stock_data)*0.80)]
        test_data = stock_data[int(len(stock_data)*0.80):]

        # Feature scaling
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_train_data = scaler.fit_transform(train_data)

        # Prepare data for LSTM
        x = []
        y = []

        # Using 100 previous days to predict the next day
        for i in range(100, scaled_train_data.shape[0]):
            x.append(scaled_train_data[i-100:i])  # 100 previous days as features
            y.append(scaled_train_data[i, 0])     # Next day's closing price as target (for regression task)

        x, y = np.array(x), np.array(y)

        # Reshape x to be 3D for LSTM input: [samples, time steps, features]
        x = np.reshape(x, (x.shape[0], x.shape[1], x.shape[2]))

        # Build and compile LSTM model
        lstm_model = self.build_and_complile_lstm(input_shape=(x.shape[1], x.shape[2]))

        # Reshape y to be 2D for the output: [samples, 1]
        y = np.reshape(y, (y.shape[0], 1))

        # Train the model
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        lstm_model.fit(x, y, epochs=self.config.lstm_epochs, batch_size=self.config.lstm_batch_size, verbose=self.config.lstm_verbose, validation_split=0.2, callbacks=[early_stopping])

        # Save the model
        if lstm_model.save(self.config.lstm_model):
            logger.info("LSTM MODEL SAVED SUCCESSFULLY.")

        # Save the test data (optional, if needed for further evaluation)
        save_as_csv(test_data, self.config.test_data_path)

        

        



