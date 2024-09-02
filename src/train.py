from models.lstm import train_lstm_model
from models.gru import train_gru_model
from models.transformer import train_transformer_model
from models.var_bar import train_var_model, train_bar_model
from data import preprocess_data
import pandas as pd
import torch

if __name__ == "__main__":
    df = preprocess_data('../data/raw/AAPL.csv')
    lstm_model = train_lstm_model(df)
    gru_model = train_gru_model(df)
    transformer_model = train_transformer_model(df)
    var_model = train_var_model(df[['Open', 'High', 'Low', 'Close']])
    bar_model = train_bar_model(df['Close'])
    
    # Saving the models
    lstm_model.save('../models/lstm_aapl.h5')
    gru_model.save('../models/gru_aapl.h5')
    torch.save(transformer_model.state_dict(), '../models/transformer_aapl.pth')
    var_model.save('../models/var_aapl.pkl')
    bar_model.save('../models/bar_aapl.pkl')
