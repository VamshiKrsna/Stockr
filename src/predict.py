import numpy as np
import pandas as pd
from keras.models import load_model
import torch
from torch import nn
from statsmodels.tsa.api import VAR
import joblib

def load_lstm_model(model_path):
    return load_model(model_path)

def load_gru_model(model_path):
    return load_model(model_path)

def load_transformer_model(model_path):
    model = nn.Transformer(d_model=64, nhead=2, num_encoder_layers=2, num_decoder_layers=2)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def load_var_model(model_path):
    return joblib.load(model_path)

def load_bar_model(model_path):
    return joblib.load(model_path)

def make_predictions_lstm(model, data):
    data = np.reshape(data, (data.shape[0], data.shape[1], 1))
    return model.predict(data)

def make_predictions_gru(model, data):
    data = np.reshape(data, (data.shape[0], data.shape[1], 1))
    return model.predict(data)

def make_predictions_transformer(model, data):
    data = torch.tensor(data).float()
    with torch.no_grad():
        output = model(data, data)
    return output.numpy()

def make_predictions_var(model, data):
    results = model.forecast(data.values[-model.k_ar:], steps=1)
    return results

def make_predictions_bar(model, data):
    results = model.forecast(data.values[-model.k_ar:], steps=1)
    return results

if __name__ == "__main__":
    # Example for making predictions on AAPL stock data
    df = pd.read_csv('data/processed/AAPL.csv')
    data = df['Scaled_Close'].values
    X = []
    for i in range(60, len(data)):
        X.append(data[i-60:i])
    X = np.array(X)

    # Load models
    lstm_model = load_lstm_model('models/lstm_aapl.h5')
    gru_model = load_gru_model('models/gru_aapl.h5')
    transformer_model = load_transformer_model('models/transformer_aapl.pth')
    var_model = load_var_model('models/var_aapl.pkl')
    bar_model = load_bar_model('models/bar_aapl.pkl')

    # Make predictions
    lstm_predictions = make_predictions_lstm(lstm_model, X)
    gru_predictions = make_predictions_gru(gru_model, X)
    transformer_predictions = make_predictions_transformer(transformer_model, X)
    var_predictions = make_predictions_var(var_model, df[['Open', 'High', 'Low', 'Close']])
    bar_predictions = make_predictions_bar(bar_model, df[['Open', 'High', 'Low', 'Close']])

    print("LSTM Predictions:", lstm_predictions[:5])
    print("GRU Predictions:", gru_predictions[:5])
    print("Transformer Predictions:", transformer_predictions[:5])
    print("VAR Predictions:", var_predictions[:5])
    print("BAR Predictions:", bar_predictions[:5])
