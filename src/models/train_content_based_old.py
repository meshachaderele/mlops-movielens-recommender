import os
import pandas as pd
import mlflow
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib


def load_items():
    # Load processed interactions
    df = pd.read_csv("data/processed/interactions.csv")

    # Keep only unique movies
    items = df[["item_id", "title"]].drop_duplicates().reset_index(drop=True)
    return items


def build_model(items, max_features=5000):
    # Build TF-IDF from movie titles for unique items
    tfidf = TfidfVectorizer(stop_words="english", max_features=max_features)
    tfidf_matrix = tfidf.fit_transform(items["title"])

    # Compute similarity between items
    sim_matrix = cosine_similarity(tfidf_matrix)

    return tfidf, sim_matrix


def save_artifacts(tfidf, sim_matrix, items):
    os.makedirs("models", exist_ok=True)
    joblib.dump(tfidf, "models/tfidf_model.joblib")
    joblib.dump(sim_matrix, "models/sim_matrix.joblib")
    items.to_csv("models/item_lookup.csv", index=False)


if __name__ == "__main__":
    mlflow.set_experiment("movielens-content-based")

    with mlflow.start_run():
        items = load_items()
        tfidf_model, sim_matrix = build_model(items)

        mlflow.log_param("model_type", "content_based")
        mlflow.log_param("tfidf_max_features", 5000)
        mlflow.log_metric("num_items", len(items))

        save_artifacts(tfidf_model, sim_matrix, items)

        # Log artifacts
        mlflow.log_artifact("models/tfidf_model.joblib")
        mlflow.log_artifact("models/sim_matrix.joblib")
        mlflow.log_artifact("models/item_lookup.csv")

        print(f"Model trained and logged. Number of items: {len(items)}")
