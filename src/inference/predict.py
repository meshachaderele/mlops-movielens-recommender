import pandas as pd
import joblib


class ContentRecommender:
    def __init__(self):
        # Load artifacts created in train_content_based.py
        self.item_lookup = pd.read_csv("models/item_lookup.csv")
        self.tfidf = joblib.load("models/tfidf_model.joblib")
        self.sim_matrix = joblib.load("models/sim_matrix.joblib")

    def recommend(self, item_title, top_k=5):
        # Find item index by fuzzy matching title
        matches = self.item_lookup[
            self.item_lookup["title"].str.contains(item_title, case=False, na=False)
        ]

        if len(matches) == 0:
            print("No item found that matches the title.")
            return []

        idx = matches.index[0]
        base_title = matches.iloc[0]["title"]

        # Get similarity scores for this item
        sim_scores = list(enumerate(self.sim_matrix[idx]))
        # Sort by similarity score
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Take top_k most similar items, skipping the item itself at position 0
        top_indices = [i for i, s in sim_scores[1 : top_k + 1]]
        recommendations = self.item_lookup.iloc[top_indices]["title"].tolist()

        print(f"Recommendations for '{base_title}':")
        return recommendations


if __name__ == "__main__":
    rec = ContentRecommender()
    recs = rec.recommend("Star Wars", top_k=5)
    for r in recs:
        print("  -", r)
