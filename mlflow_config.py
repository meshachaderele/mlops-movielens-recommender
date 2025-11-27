import mlflow

def configure_mlflow():
    # Use a local sqlite database for tracking and model registry
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    return mlflow
