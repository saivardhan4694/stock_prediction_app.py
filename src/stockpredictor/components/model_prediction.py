from src.stockpredictor.utils.common import save_as_csv
from src.stockpredictor.logging.coustom_log import logger
from src.stockpredictor.entity import ModelPredictionConfig
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta

class ModelPrediction:
    def __init__(self, config: ModelPredictionConfig):
        self.config = config

    def predict_next_30_days(self, model, last_known_data, scaler, n_days=30):
        predictions = []
        current_input = last_known_data  # Start with the last known data (the most recent 100 days)

        # List of feature names in the same order as your model's output
        feature_names = ['Close', 'moving_avg_10', 'moving_avg_30', 'rsi', 'volatility_5', 
                        'volatility_10', 'atr', 'price_change_pct', 'day_of_week', 'month']

        for _ in range(n_days):
            # Predict the next day using the current input
            predicted_scaled = model.predict(current_input)  # This will output the predicted values for the next day (10 features)
            
            # Extract the predicted 10 feature values
            predicted_values = predicted_scaled[0, :]  # Get the first row, which contains the predicted values for all 10 features
            
            # Append only the predicted values (numerical) to the predictions list
            predictions.append(predicted_values)  # predictions is now a list of numerical arrays
            
            # Update the input for the next prediction by appending the predicted values
            predicted_scaled_reshaped = predicted_values.reshape(1, 1, 10)  # Reshape to (1, 1, 10)
            current_input = np.append(current_input[:, 1:, :], predicted_scaled_reshaped, axis=1)

        # Convert predictions list into a numpy array and inverse scale the predicted values
        predictions = np.array(predictions)

        # Now inverse transform the numerical array (not a dict)
        predictions = scaler.inverse_transform(predictions)

        return predictions


    def forecast_30_days(self):
        # Read the data from the CSV file
        data = pd.read_csv(self.config.forecast_input)

        # Feature selection: Using 'Close' and technical indicators
        features = ['Close', 'moving_avg_10', 'moving_avg_30', 'rsi', 'volatility_5', 'volatility_10', 
                    'atr', 'price_change_pct', 'day_of_week', 'month']
        
        # Extract features and target variable (closing price)
        stock_data = data[features]
        
        # Scale the features
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(stock_data)

        # Prepare the last 100 days of data as the input for prediction
        last_known_data = scaled_data[-100:].reshape(1, 100, len(features))  # Reshape for LSTM input

        # Load the saved LSTM model
        model = load_model(self.config.model)

        # Make predictions for the next 30 days
        predictions = self.predict_next_30_days(model, last_known_data, scaler, n_days=30)

         # Generate future dates for the predictions
        last_date = pd.to_datetime(data['Date'].iloc[-1])  # Get the last date in the dataset
        future_dates = [last_date + timedelta(days=i) for i in range(1, 31)]  # Generate the next 30 days

        # Create a DataFrame to save predictions with dates for each of the 10 features
        predictions_df = pd.DataFrame(predictions, columns=['Close', 'moving_avg_10', 'moving_avg_30', 'rsi', 
                                                           'volatility_5', 'volatility_10', 'atr', 'price_change_pct', 
                                                           'day_of_week', 'month'])
        predictions_df['Date'] = future_dates

        # Save predictions to a CSV file
        predictions_df.to_csv(self.config.predictions_output, index=False)

        # Log success
        logger.info(f"Predictions saved successfully to {self.config.predictions_output}")

        # Plot the predictions for the 'Close' price
        plt.figure(figsize=(10, 6))
        plt.plot(future_dates, predictions_df['Close'], label='Forecasted Closing Prices', color='red')
        plt.title('Stock Price Forecast for the Next 30 Days')
        plt.xlabel('Days')
        plt.ylabel('Stock Price')
        plt.xticks(rotation=45)
        plt.legend()
        plt.show()

        # Save the plot as an image file
        plot_image_path = self.config.plot_output  
        plt.savefig(plot_image_path, bbox_inches='tight')  
        logger.info(f"Plot saved successfully as {plot_image_path}")
