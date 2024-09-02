# pipeline.py
import os
import subprocess
import mlflow

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

def preprocess_data():
    """
    Run the feature engineering script to preprocess data.
    """
    print("Preprocessing data...")
    run_command("python src/features/feature_engineering.py")

def train_models():
    """
    Run the training script to train all models.
    """
    print("Training models...")
    run_command("python src/train.py")

def evaluate_models():
    """
    Run the evaluation script to evaluate all trained models.
    """
    print("Evaluating models...")
    run_command("python src/evaluate.py")

def log_pipeline_to_mlflow():
    """
    Log the pipeline steps and metrics to MLFlow.
    """
    mlflow.set_experiment("Stock Forecasting Pipeline")
    with mlflow.start_run():
        mlflow.log_param("Stage", "Data Preprocessing")
        preprocess_data()
        mlflow.log_param("Stage", "Model Training")
        train_models()
        mlflow.log_param("Stage", "Model Evaluation")
        evaluate_models()
        mlflow.log_artifacts("models/")  # Log trained models

if __name__ == "__main__":
    print("Running the Stock Forecasting Pipeline...")
    log_pipeline_to_mlflow()
    print("Pipeline execution completed.")
