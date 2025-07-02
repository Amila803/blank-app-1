# data/train_cost_rf_tuned.py

import re
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ─── Helpers ────────────────────────────────────────────────────────────────
def clean_cost(val):
    """Strip non-digits, convert to float, coerce errors to NaN."""
    if pd.isna(val):
        return np.nan
    if isinstance(val, str):
        nums = re.sub(r"[^\d.]", "", val)
        return float(nums) if nums else np.nan
    return float(val)

# ─── Load & Clean ───────────────────────────────────────────────────────────
def load_data(csv_path: Path):
    df = pd.read_csv(csv_path)
    # Drop totally empty rows
    df = df.dropna(how="all")

    # Clean numeric costs
    df["Accommodation cost"] = df["Accommodation cost"].apply(clean_cost)
    df["Transportation cost"] = df["Transportation cost"].apply(clean_cost)
    df["Total cost"] = df["Accommodation cost"] + df["Transportation cost"]

    # Parse dates
    df["Start date"] = pd.to_datetime(df["Start date"], errors="coerce")
    df["End date"]   = pd.to_datetime(df["End date"],   errors="coerce")

    # Drop rows missing critical data
    df = df.dropna(subset=[
        "Destination", "Start date", "End date", "Total cost"
    ])

    return df

# ─── Feature Engineering ───────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame):
    # Extract city only from "City, Country"
    df["City"] = df["Destination"].str.split(",").str[0].str.strip()

    # Temporal features
    df["Year"]      = df["Start date"].dt.year
    df["Month"]     = df["Start date"].dt.month
    df["DayOfWeek"] = df["Start date"].dt.dayofweek
    df["IsWeekend"] = df["DayOfWeek"].isin([5,6]).astype(int)
    df["Duration"]  = (df["End date"] - df["Start date"]).dt.days.clip(lower=1)

    # Season (1=winter,2=spring,3=summer,4=fall if northern hemisphere)
    df["Season"] = ((df["Month"] % 12) // 3 + 1).astype(int)

    # Group low-freq cities into “Other”
    city_counts = df["City"].value_counts()
    mask = df["City"].isin(city_counts[city_counts < 5].index)
    df.loc[mask, "City"] = "Other"

    # Finalize features & target
    feats = [
        "City", "Accommodation type", "Transportation type",
        "Duration", "Year", "Month", "DayOfWeek", "IsWeekend", "Season"
    ]
    target = "Total cost"

    df = df.dropna(subset=feats + [target])
    X = df[feats]
    y = df[target].astype(float)
    return X, y

# ─── Train & Tune ──────────────────────────────────────────────────────────
def train_and_tune(csv_path: Path, out_dir: Path):
    # Load & preprocess
    raw = load_data(csv_path)
    X, y = engineer_features(raw)

    # Build preprocessing + model pipeline
    cat_features = ["City", "Accommodation type", "Transportation type"]
    num_features = [
        "Duration", "Year", "Month", "DayOfWeek", "IsWeekend", "Season"
    ]

    preprocessor = ColumnTransformer([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_features),
        ("passthru", "passthrough", num_features)
    ])

    pipeline = Pipeline([
        ("prep", preprocessor),
        ("rf",   RandomForestRegressor(random_state=42, n_jobs=-1))
    ])

    # Hyperparameter distributions
    param_dist = {
        "rf__n_estimators":      [100, 200, 500, 1000],
        "rf__max_depth":         [None, 10, 20, 30, 50],
        "rf__min_samples_split": [2, 5, 10],
        "rf__min_samples_leaf":  [1, 2, 4],
        "rf__bootstrap":         [True, False]
    }

    # 5-fold CV
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=30,
        cv=cv,
        scoring="neg_mean_absolute_error",
        verbose=2,
        n_jobs=-1,
        random_state=42
    )

    # Fit
    search.fit(X, y)
    best = search.best_estimator_

    # Evaluate on full data
    preds = best.predict(X)
    mae  = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    r2   = r2_score(y, preds)

    print(f"Best RF params: {search.best_params_}")
    print(f"In-sample MAE:  {mae:,.2f}")
    print(f"In-sample RMSE: {rmse:,.2f}")
    print(f"In-sample R²:   {r2:.4f}")

    # Save model
    out_dir.mkdir(exist_ok=True, parents=True)
    model_path = out_dir / "travel_cost_rf_tuned.joblib"
    joblib.dump(best, model_path)
    print("✅ Tuned model saved to", model_path)
    return model_path

# ─── Entry Point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    CSV  = ROOT / "data" / "Travel_details_dataset.csv"
    OUT  = ROOT / "models"
    train_and_tune(CSV, OUT)
