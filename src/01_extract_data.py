from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo


RAW_DATA_PATH = Path("data/raw/occupancy_raw.csv")


def extract_data():
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    dataset = fetch_ucirepo(id=357)

    features = dataset.data.features
    target = dataset.data.targets

    df = pd.concat([features, target], axis=1)

    df.to_csv(RAW_DATA_PATH, index=False)

    print(f"Raw data saved to: {RAW_DATA_PATH}")
    print(f"Dataset shape: {df.shape}")


if __name__ == "__main__":
    extract_data()
