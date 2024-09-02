# retrain.py
import os
import subprocess
import mlflow
from datetime import datetime

def run_command(command):
    """
    Utility function to run shell commands.
    """
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(f"Error running command: {command}\n{stderr.decode('utf-8')}")
    else:
        print(stdout.decode('utf-8'))

def update_data_with_dvc():
    """
    Pull the latest data version using DVC.
    """
    print("Updating data using DVC...")
    run_command("dvc pull")

def retrain_models():
    """
    Retrain models with updated data.
    """
    print("Retraining models...")
    run_command("python src/train.py")

def evaluate_updated_models():
    """
    Evaluate the retrained models.
    """
    print("Evaluating retrained models...")
    run_command("python src/evaluate.py")

def log_retrain_to_mlflow():
    """
    Log the retraining steps and evaluation metrics to MLFlow.
    """
    mlflow.set_experiment("Stock Forecasting Retraining")
    with mlflow.start_run(run_name=f"Retrain-{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"):
        mlflow.log_param("Stage", "Data Update")
        update_data_with_dvc()
        mlflow.log_param("Stage", "Model Retraining")
        retrain_models()
        mlflow.log_param("Stage", "Model Evaluation")
        evaluate_updated_models()
        mlflow.log_artifacts("models/")  # Log retrained models

if __name__ == "__main__":
    print("Starting the retraining process...")
    log_retrain_to_mlflow()
    print("Retraining process completed.")
