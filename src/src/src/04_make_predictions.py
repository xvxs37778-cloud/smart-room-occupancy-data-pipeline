from pathlib import Path
import pandas as pd
import joblib


DATA_PATH = Path("data/processed/occupancy_cleaned.csv")
MODEL_PATH = Path("models/occupancy_model.pkl")
OUTPUT_PATH = Path("data/output/occupancy_predictions.csv")


def make_predictions():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    model_package = joblib.load(MODEL_PATH)
    model = model_package["model"]
    features = model_package["features"]

    X = df[features]

    df["predicted_occupancy"] = model.predict(X)
    df["occupancy_probability"] = model.predict_proba(X)[:, 1]

    df["prediction_label"] = df["predicted_occupancy"].map({
        0: "Not Occupied",
        1: "Occupied"
    })

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Predictions saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    make_predictions()
