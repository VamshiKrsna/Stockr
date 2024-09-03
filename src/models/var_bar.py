import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.statespace.sarimax import SARIMAX

def train_var_model(df, maxlags=15):
    model = VAR(df)
    results = model.fit(maxlags)
    return results

def train_bar_model(df):
    # Placeholder for BAR (Bayesian Autoregression)
    model = SARIMAX(df, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    results = model.fit()
    return results

if __name__ == "__main__":
    df = pd.read_csv('../data/processed/AAPL.csv')
    var_model = train_var_model(df[['Open', 'High', 'Low', 'Close']])
    bar_model = train_bar_model(df['Close'])
