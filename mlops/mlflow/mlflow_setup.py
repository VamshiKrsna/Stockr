import mlflow
import mlflow.keras
from lstm import train_lstm_model

if __name__ == "__main__":
    mlflow.set_experiment("stockr_lstm")
    with mlflow.start_run():
        model = train_lstm_model()
        mlflow.keras.log_model(model, "model")
