from pathlib import Path
import pandas as pd
import numpy as np


RAW_DATA_PATH = Path("data/raw/occupancy_raw.csv")
PROCESSED_DATA_PATH = Path("data/processed/occupancy_cleaned.csv")


def clean_column_names(df):
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )
    return df


def add_time_features(df):
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df = df.sort_values("date")

        df["hour"] = df["date"].dt.hour
        df["day_of_week"] = df["date"].dt.dayofweek
        df["is_working_hours"] = df["hour"].between(8, 18).astype(int)

    return df


def add_sensor_features(df):
    if "co2" in df.columns:
        df["co2_rolling_mean_5"] = df["co2"].rolling(window=5, min_periods=1).mean()
        df["co2_change"] = df["co2"].diff().fillna(0)

    if "light" in df.columns:
        df["light_rolling_mean_5"] = df["light"].rolling(window=5, min_periods=1).mean()
        df["is_light_on"] = (df["light"] > 100).astype(int)

    if "temperature" in df.columns:
        df["temperature_change"] = df["temperature"].diff().fillna(0)

    if "humidity" in df.columns:
        df["humidity_change"] = df["humidity"].diff().fillna(0)

    return df


def handle_outliers(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns

    for col in numeric_columns:
        lower_limit = df[col].quantile(0.01)
        upper_limit = df[col].quantile(0.99)
        df[col] = df[col].clip(lower=lower_limit, upper=upper_limit)

    return df


def transform_data():
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_DATA_PATH)
    df = clean_column_names(df)

    df = df.drop_duplicates()
    df = df.dropna()

    df = add_time_features(df)
    df = add_sensor_features(df)
    df = handle_outliers(df)

    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print(f"Processed data saved to: {PROCESSED_DATA_PATH}")
    print(f"Processed shape: {df.shape}")
    print("Columns:")
    print(df.columns.tolist())


if __name__ == "__main__":
    transform_data()
