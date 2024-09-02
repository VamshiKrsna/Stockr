import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class TransformerTimeSeries(nn.Module):
    def __init__(self, input_dim, nhead, num_layers, dim_feedforward):
        super(TransformerTimeSeries, self).__init__()
        self.encoder = nn.Linear(input_dim, dim_feedforward)
        self.transformer = nn.Transformer(
            d_model=dim_feedforward, nhead=nhead, num_encoder_layers=num_layers, num_decoder_layers=num_layers
        )
        self.decoder = nn.Linear(dim_feedforward, 1)

    def forward(self, src, tgt):
        src = self.encoder(src)
        tgt = self.encoder(tgt)
        output = self.transformer(src, tgt)
        return self.decoder(output)

def train_transformer_model(data, epochs=50):
    X_train, y_train = data
    model = TransformerTimeSeries(input_dim=1, nhead=2, num_layers=2, dim_feedforward=64)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(torch.tensor(X_train).float(), torch.tensor(X_train).float())
        loss = criterion(output, torch.tensor(y_train).float().unsqueeze(1))
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')
    return model

if __name__ == "__main__":
    df = pd.read_csv('../data/processed/AAPL.csv')
    data = df['Scaled_Close'].values
    X, y = [], []
    for i in range(60, len(data)):
        X.append(data[i-60:i])
        y.append(data[i])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    train_transformer_model((X, y))
