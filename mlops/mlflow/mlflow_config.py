import mlflow

def setup_mlflow():
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Stock Forecasting")
    print("MLFlow tracking URI set and experiment initialized.")

if __name__ == "__main__":
    setup_mlflow()
