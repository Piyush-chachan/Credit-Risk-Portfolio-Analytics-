"""`python -m src.build_assets` - build the clean dataset, train the model, run anomaly detection, write insights."""
from .assets import build_all

if __name__ == "__main__":
    build_all(save=True, verbose=True)
