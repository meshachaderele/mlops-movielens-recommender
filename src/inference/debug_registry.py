from mlflow.tracking import MlflowClient
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # goes from models -> src -> project
sys.path.append(str(ROOT))
from mlflow_config import configure_mlflow

from mlflow_config import configure_mlflow


def main():
    configure_mlflow()
    client = MlflowClient()

    model_name = "movie-recommender"

    print(f"=== Registered model: {model_name} ===")
    try:
        info = client.get_registered_model(model_name)
    except Exception as e:
        print("Could not find registered model:", e)
        return

    print("Model info:", info.name)
    print("Versions:")
    for mv in client.search_model_versions(f"name = '{model_name}'"):
        print(
            f"  version={mv.version}, aliases={mv.aliases}, tags={mv.tags}"
        )


if __name__ == "__main__":
    main()

