import pickle
from pathlib import Path


def save_model(path, model, scaler):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump({"model": model, "scaler": scaler}, f)


def load_model(path):
    with Path(path).open("rb") as f:
        data = pickle.load(f)
    return data["model"], data["scaler"]
