from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


DATA_PATH = Path("data/processed/occupancy_cleaned.csv")
MODEL_PATH = Path("models/occupancy_model.pkl")
REPORT_PATH = Path("reports/model_report.txt")


def train_model():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    target_column = "occupancy"

    if target_column not in df.columns:
        raise ValueError("Target column 'occupancy' was not found in the dataset.")

    drop_columns = [target_column]

    if "date" in df.columns:
        drop_columns.append("date")

    X = df.drop(columns=drop_columns)
    y = df[target_column]

    X = X.select_dtypes(include=["number"])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    joblib.dump(
        {
            "model": model,
            "features": X.columns.tolist()
        },
        MODEL_PATH
    )

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        file.write("Smart Room Occupancy Prediction Model\n")
        file.write("=" * 45)
        file.write("\n\n")
        file.write(f"Accuracy: {accuracy:.4f}\n\n")
        file.write(report)

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Report saved to: {REPORT_PATH}")
    print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    train_model()
