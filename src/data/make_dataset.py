import os
import pandas as pd


def load_raw_movielens():
    ratings_path = os.path.join("data", "raw", "ml-100k", "u.data")
    item_path = os.path.join("data", "raw", "ml-100k", "u.item")

    ratings = pd.read_csv(
        ratings_path,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        encoding="latin-1"
    )

    items = pd.read_csv(
        item_path,
        sep="|",
        header=None,
        encoding="latin-1",
        usecols=[0, 1, 2],
        names=["item_id", "title", "release_date"]
    )

    df = ratings.merge(items, on="item_id", how="left")

    return df


def preprocess(df):
    # Keep only useful columns
    df = df[["user_id", "item_id", "rating", "title"]]

    # Remove missing titles if any
    df = df.dropna(subset=["title"])

    return df


def save_processed(df):
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/interactions.csv", index=False)


if __name__ == "__main__":
    df = load_raw_movielens()
    df = preprocess(df)
    save_processed(df)
    print("Saved processed data to data/processed/interactions.csv")
