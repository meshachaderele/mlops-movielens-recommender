import mlflow

def configure_mlflow():
    # Use a local sqlite database for tracking and model registry
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    #mlflow.set_tracking_uri("http://127.0.0.1:5000")
    return mlflow
