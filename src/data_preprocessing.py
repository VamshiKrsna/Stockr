# Helper file to preprocess data into train and test sets (not very useful, anyways)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(file_path):
    df = pd.read_csv(file_path,index_col=0)
    df = df.dropna()

    x = df[["Open","High","Low","Volume"]]
    y = df["Close"]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)

    scaler = StandardScaler()
    scaled_X_train = scaler.fit_transform(X_train)
    scaled_X_test = scaler.transform(X_test)

    return scaled_X_train, y_train, scaled_X_test, y_test

if __name__ == "__main__":
    for stock in ["AAPL","MSFT","GOOGL","AMZN","TSLA"]:
        file_path = f"../data/{stock}.csv"
        X_train, X_test, y_train, y_test = preprocess_data(file_path)
        print(f"Preprocessed data for {stock}")
        # print(X_train, X_test, y_train, y_test)