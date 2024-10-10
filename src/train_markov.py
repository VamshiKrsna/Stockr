import os
import warnings
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

data_dir = "../data/"
model_dir = "../models"

os.makedirs(model_dir, exist_ok=True)

def train_markov_chain(df, stock_name, forecast_days=30):
    """
    Trains a Markov Chain model on the given data and forecasts the next `forecast_days` days.

    Parameters:
        df (pd.DataFrame): The input dataframe with datetime index and numeric columns.
        stock_name (str): The name of the stock to use in saving the model files.
        forecast_days (int): The number of days to forecast. Default is 30.

    Returns:
        None: Displays the plot of actual vs forecasted values, prints the MSE, and saves the model.
    """
    # Selecting numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

    # Scaling the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[numeric_cols])

    # Splitting the data into training and testing sets
    train_data, test_data = scaled_data[:-forecast_days], scaled_data[-forecast_days:]

    # Create a Markov Chain model
    transition_matrix = np.zeros((len(train_data), len(train_data)))
    for i in range(len(train_data) - 1):
        transition_matrix[i, np.argmax(train_data[i + 1])] += 1

    # Normalize the transition matrix
    transition_matrix /= transition_matrix.sum(axis=1, keepdims=True)

    # Forecast the next `forecast_days` using the Markov Chain model
    forecasted_values = np.zeros(forecast_days)
    current_state = train_data[-1]
    for i in range(forecast_days):
        next_state = np.random.choice(len(train_data), p=transition_matrix[np.argmax(current_state)])
        forecasted_values[i] = train_data[next_state]
        current_state = train_data[next_state]

    # Scale back to original values
    forecasted_values_rescaled = scaler.inverse_transform(forecasted_values.reshape(-1, 1))
    actual_values_rescaled = scaler.inverse_transform(test_data[:, -1].reshape(-1, 1))

    # Evaluate the model
    mse = mean_squared_error(actual_values_rescaled, forecasted_values_rescaled)
    print(f'MSE for {stock_name}: ', mse)

# Save the model
    model_path = os.path.join(model_dir, f"{stock_name}_markov_chain_model.joblib")
    joblib.dump((scaler, transition_matrix), model_path)
    print(f"Model saved for {stock_name} at {model_path}.")

    # Plot actual vs forecasted values
    plt.figure(figsize=(14, 7))
    plt.plot(range(len(actual_values_rescaled)), actual_values_rescaled, label='Actual', color='blue')
    plt.plot(range(len(forecasted_values_rescaled)), forecasted_values_rescaled, label='Forecasted', color='orange')
    plt.title(f'Actual vs Forecasted Prices for {stock_name} - Last {forecast_days} Days')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

    for stock in stocks:
        file_path = os.path.join(data_dir, f"{stock}.csv")
        data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')

        print(f"Training models for {stock}...")

        # Train Markov Chain models for each stock and save them in the models directory
        train_markov_chain(data, stock_name=stock, forecast_days=30)

        print(f"Models for {stock} saved successfully.")