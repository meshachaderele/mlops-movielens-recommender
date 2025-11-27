from mlflow.tracking import MlflowClient
import mlflow
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]  # project root
sys.path.append(str(ROOT))
from mlflow_config import configure_mlflow


def main():
    configure_mlflow()
    client = MlflowClient()

    model_name = "movie-recommender"
    
    run_id = "74d0cc82f3f941809e2ef7dcaa0daf0c"


    source = f"runs:/{run_id}/model"
    print(f"Registering model version from: {source}")

    # 2. Ensure registered model exists
    try:
        client.get_registered_model(model_name)
        print(f"Registered model '{model_name}' already exists.")
    except Exception:
        print(f"Registered model '{model_name}' does not exist. Creating it.")
        client.create_registered_model(model_name)

    # 3. Create a new model version
    mv = client.create_model_version(
        name=model_name,
        source=source,
        run_id=run_id,
    )
    print(f"Created model version: {mv.version}")

    # 4. Optional - set alias `production` to this version
    alias = "production"
    client.set_registered_model_alias(
        name=model_name,
        alias=alias,
        version=mv.version,
    )
    print(f"Set alias '{alias}' -> {model_name} v{mv.version}")

    # 5. Show all versions
    print(f"\n=== Registered model: {model_name} ===")
    info = client.get_registered_model(model_name)
    print("Model info:", info.name)
    print("Versions:")
    for mv in client.search_model_versions(f"name = '{model_name}'"):
        print(
            f"  version={mv.version}, aliases={mv.aliases}, tags={mv.tags}"
        )


if __name__ == "__main__":
    main()
