import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

def load_data():
    # path relative to project root
    ratings_path = os.path.join("data", "raw", "ml-100k", "u.data")
    df = pd.read_csv(
        ratings_path,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        encoding="latin-1"
    )
    return df

def build_baseline(df):
    #predict global mean rating
    global_mean = df["rating"].mean()
    return global_mean

def evaluate_baseline(df, global_mean):
    # Split to demonstrate evaluation
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    y_true = test["rating"].values
    y_pred = [global_mean] * len(y_true)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    return rmse

if __name__ == "__main__":
    mlflow.set_experiment("movielens-baseline")

    with mlflow.start_run():
        df = load_data()
        global_mean = build_baseline(df)
        rmse = evaluate_baseline(df, global_mean)

        # log parameters, metrics and model
        mlflow.log_param("model_type", "global_mean")
        mlflow.log_metric("rmse", rmse)

        # store model as a simple artifact
        mlflow.log_artifact("requirements.txt")

        print(f"Baseline RMSE: {rmse}")
