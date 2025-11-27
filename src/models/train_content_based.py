import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import mlflow
import mlflow.pyfunc

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # goes from models -> src -> project
sys.path.append(str(ROOT))

from mlflow_config import configure_mlflow


class MovieRecommenderModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import pandas as pd
        import joblib

        self.item_lookup = pd.read_csv(context.artifacts["item_lookup"])
        self.sim_matrix = joblib.load(context.artifacts["sim_matrix"])

    def predict(self, context, model_input):
        """
        model_input: pandas DataFrame with a column 'title' and optional 'top_k'
        Returns: list of lists of recommended titles per input row.
        """
        results = []

        for _, row in model_input.iterrows():
            query = str(row.get("title", ""))
            top_k = int(row.get("top_k", 5))

            matches = self.item_lookup[
                self.item_lookup["title"].str.contains(query, case=False, na=False)
            ]

            if len(matches) == 0:
                results.append([])
                continue

            idx = matches.index[0]

            sim_scores = list(enumerate(self.sim_matrix[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

            top_indices = [i for i, s in sim_scores[1 : top_k + 1]]
            recommended = self.item_lookup.iloc[top_indices]["title"].tolist()

            results.append(recommended)

        return results


def load_items():
    df = pd.read_csv("data/processed/interactions.csv")
    items = df[["item_id", "title"]].drop_duplicates().reset_index(drop=True)
    return items


def build_item_similarity(items, max_features=5000):
    tfidf = TfidfVectorizer(stop_words="english", max_features=max_features)
    tfidf_matrix = tfidf.fit_transform(items["title"])
    sim_matrix = cosine_similarity(tfidf_matrix)
    return tfidf, sim_matrix


def main():
    configure_mlflow()
    experiment_name = "movielens-content-based"
    model_name = "movie-recommender"

    mlflow.set_experiment(experiment_name)

    items = load_items()
    tfidf, sim_matrix = build_item_similarity(items)

    os.makedirs("models", exist_ok=True)
    joblib.dump(sim_matrix, "models/sim_matrix.joblib")
    items.to_csv("models/item_lookup.csv", index=False)

    with mlflow.start_run() as run:
        mlflow.log_param("model_type", "content_based")
        mlflow.log_param("tfidf_max_features", 5000)
        mlflow.log_metric("num_items", len(items))

        artifacts = {
            "sim_matrix": "models/sim_matrix.joblib",
            "item_lookup": "models/item_lookup.csv",
        }

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=MovieRecommenderModel(),
            artifacts=artifacts,
            registered_model_name=model_name,
        )

        print(f"Logged model to run_id={run.info.run_id}")


if __name__ == "__main__":
    main()
