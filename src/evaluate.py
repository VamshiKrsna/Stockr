import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(true_values, predicted_values, model_name):
    mse = mean_squared_error(true_values, predicted_values)
    mae = mean_absolute_error(true_values, predicted_values)
    r2 = r2_score(true_values, predicted_values)

    print(f"Evaluation Metrics for {model_name}:")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R-Squared (R2): {r2}")
    print("=" * 50)
    return mse, mae, r2

if __name__ == "__main__":
    # Load the true data
    df = pd.read_csv('data/processed/AAPL.csv')
    true_values = df['Scaled_Close'].values[60:]  # True values starting after the training window

    # Load predictions from predict.py results
    lstm_predictions = np.load('predictions/lstm_predictions.npy')
    gru_predictions = np.load('predictions/gru_predictions.npy')
    transformer_predictions = np.load('predictions/transformer_predictions.npy')
    var_predictions = np.load('predictions/var_predictions.npy')
    bar_predictions = np.load('predictions/bar_predictions.npy')

    # Evaluate models
    evaluate_model(true_values, lstm_predictions, "LSTM Model")
    evaluate_model(true_values, gru_predictions, "GRU Model")
    evaluate_model(true_values, transformer_predictions, "Transformer Model")
    evaluate_model(true_values, var_predictions[:, 3], "VAR Model")  # Assuming 'Close' is at index 3
    evaluate_model(true_values, bar_predictions[:, 3], "BAR Model")
