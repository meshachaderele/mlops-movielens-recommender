import pandas as pd
import mlflow
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # goes from models -> src -> project
sys.path.append(str(ROOT))
from mlflow_config import configure_mlflow


model_name = "movie-recommender"
model_version = 1

# Load the model from the Model Registry
model_uri = f"models:/{model_name}/{model_version}"


def main():
    configure_mlflow()

    # Load the current production model from the registry
    model_uri = f"models:/{model_name}/{model_version}"

    model = mlflow.pyfunc.load_model(model_uri)

    # Prepare a small test input
    input_df = pd.DataFrame(
        [
            {"title": "Philadelphia Story", "top_k": 5},
            {"title": "Toy Story", "top_k": 3},
        ]
    )

    preds = model.predict(input_df)
    for query, recs in zip(input_df["title"], preds):
        print(f"Recommendations for '{query}':")
        for r in recs:
            print("  -", r)


if __name__ == "__main__":
    main()
