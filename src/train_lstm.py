import os
import warnings
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

data_dir = "../data/"
model_dir = "../models"

os.makedirs(model_dir, exist_ok=True)

def evaluate_lstm(df, stock_name, target_col_index = 3, forecast_days = 30, epochs = 20): 
    """
    Trains and evaluates LSTM Model on given data, by considering first n-30 as training data and 
    tests its predictions on last 30 data values by MSE.

    Parameters:
        df (pd.DataFrame): The input dataframe with datetime index and numeric columns.
        forecast_days (int) : used to choose training and testing data values. Default is 30 days.
        target_col_index (int): The index of the target column ('Close' price typically).
        epochs (int): The number of epochs for training the LSTM model. Default is 20
        stock_name (str): name of the stock being forecasted and evaluated.

    Returns:
        MSE : Mean Squared Error of the LSTM Model.
    """
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[numeric_cols])

    train_data, test_data = scaled_data[:-forecast_days], scaled_data[-(forecast_days + 60):]

    def create_dataset(data, time_step=1):
        X, y = [], []
        for i in range(len(data) - time_step):
            X.append(data[i:(i + time_step), :])
            y.append(data[i + time_step, target_col_index])  
        return np.array(X), np.array(y)

    time_step = 60  
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    if X_test.size == 0 or y_test.size == 0:
        print("Error: Not enough data points for testing. Adjust the forecast period or time step.")
        return
    
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=1)

    test_predict = model.predict(X_test)

    test_predict_rescaled = scaler.inverse_transform(
        np.hstack((np.zeros((test_predict.shape[0], X_test.shape[2] - 1)), test_predict)))
    y_test_rescaled = scaler.inverse_transform(
        np.hstack((np.zeros((y_test.shape[0], X_test.shape[2] - 1)), y_test.reshape(-1, 1))))

    mse = mean_squared_error(y_test_rescaled[:, -1], test_predict_rescaled[:, -1])
    print(f'MSE for {stock_name}: ', mse)
    return mse


    # Plot actual vs forecasted values
    # plt.figure(figsize=(14, 7))
    # plt.plot(range(len(y_test_rescaled)), y_test_rescaled[:, -1], label='Actual', color='blue')
    # plt.plot(range(len(test_predict_rescaled)), test_predict_rescaled[:, -1], label='Forecasted', color='orange')
    # plt.title(f'Actual vs Forecasted Prices for {stock_name} - Last {forecast_days} Days')
    # plt.xlabel('Days')
    # plt.ylabel('Price')
    # plt.legend()
    # plt.show()

def train_lstm(df, stock_name, target_col_index=3, forecast_days=30, epochs=20):
    """
    Trains an LSTM model on the given data and forecasts the next `forecast_days` days.

    Parameters:
        df (pd.DataFrame): The input dataframe with datetime index and numeric columns.
        stock_name (str): The name of the stock to use in saving the model files.
        target_col_index (int): The index of the target column ('Close' price typically).
        forecast_days (int): The number of days to forecast. Default is 30.
        epochs (int): The number of epochs for training the LSTM model. Default is 20.

    Returns:
        None: Displays the plot of actual vs forecasted values, prints the MSE, and saves the model.
    """
    # Selecting numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

    # Scaling the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[numeric_cols])

    # Splitting the data into training and testing sets
    train_data, test_data = scaled_data[:-forecast_days], scaled_data[-(forecast_days + 60):]

    # Function to create input (X) and output (y) datasets
    def create_dataset(data, time_step=1):
        X, y = [], []
        for i in range(len(data) - time_step):
            X.append(data[i:(i + time_step), :])
            y.append(data[i + time_step, target_col_index])  # Using the target column index
        return np.array(X), np.array(y)

    time_step = 60  # Number of time steps to consider for each sample
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    # Check if X_test and y_test are non-empty and have the correct shape before reshaping
    if X_test.size == 0 or y_test.size == 0:
        print("Error: Not enough data points for testing. Adjust the forecast period or time step.")
        return
    
    # Reshape the data for the LSTM
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

    # Create the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=1)

    # Define the path to save the model
    model_path = os.path.join(model_dir, f"{stock_name}_lstm_model.h5")
    model.save(model_path)
    print(f"Model saved for {stock_name} at {model_path}.")

    # Forecast the next `forecast_days` using the last available data
    test_predict = model.predict(X_test)

    # Scale back to original values
    test_predict_rescaled = scaler.inverse_transform(
        np.hstack((np.zeros((test_predict.shape[0], X_test.shape[2] - 1)), test_predict)))
    y_test_rescaled = scaler.inverse_transform(
        np.hstack((np.zeros((y_test.shape[0], X_test.shape[2] - 1)), y_test.reshape(-1, 1))))

    # Evaluate the model
    mse = mean_squared_error(y_test_rescaled[:, -1], test_predict_rescaled[:, -1])
    print(f'MSE for {stock_name}: ', mse)
    return mse


if __name__ == "__main__":
    stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    acc_dict = {}

    for stock in stocks:
        file_path = os.path.join(data_dir, f"{stock}.csv")
        data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')

        print(f"Training models for {stock}...")

        # Train LSTM models for each stock and save them in the models directory
        mse = train_lstm(data, stock_name=stock, forecast_days=30, epochs=20)
        # mse = evaluate_lstm(data,stock_name=stock)

        acc_dict[stock] = mse
        print(f"Models for {stock} saved successfully.")

print(acc_dict)