# streamlit_app.py

import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import re
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# ─── Helpers ────────────────────────────────────────────────────────────────
def clean_cost(val):
    if pd.isna(val): return np.nan
    if isinstance(val, str):
        nums = re.sub(r"[^\d.]", "", val)
        return float(nums) if nums else np.nan
    return float(val)

@st.cache_data(show_spinner=False)
def load_data(csv_path: Path):
    df = pd.read_csv(csv_path)
    df = df.dropna(how="all")
    df["Accommodation cost"]   = df["Accommodation cost"].apply(clean_cost)
    df["Transportation cost"]  = df["Transportation cost"].apply(clean_cost)
    df["Total cost"]           = df["Accommodation cost"] + df["Transportation cost"]
    df["Start date"]           = pd.to_datetime(df["Start date"], errors="coerce")
    df["End date"]             = pd.to_datetime(df["End date"],   errors="coerce")
    df = df.dropna(subset=["City","Start date","End date","Total cost"], errors="ignore")
    # If your column is named "Destination", extract city:
    if "Destination" in df.columns:
        df["City"] = df["Destination"].str.split(",").str[0].str.strip()
    df["Year"]      = df["Start date"].dt.year
    df["Month"]     = df["Start date"].dt.month
    df["DayOfWeek"] = df["Start date"].dt.dayofweek
    df["IsWeekend"] = df["DayOfWeek"].isin([5,6]).astype(int)
    df["Duration"]  = (df["End date"] - df["Start date"]).dt.days.clip(lower=1)
    df["Season"]    = ((df["Month"] % 12)//3 + 1).astype(int)
    df = df.dropna(subset=["City","Accommodation cost","Transportation cost","Duration"])
    return df

@st.cache_resource(show_spinner=False)
def get_model(df: pd.DataFrame, model_path: Path):
    model_path.parent.mkdir(exist_ok=True, parents=True)
    if model_path.exists():
        return joblib.load(model_path)
    # Prepare features/target
    features = ["City", "Accommodation type", "Transportation type",
                "Duration", "Year", "Month", "DayOfWeek", "IsWeekend", "Season"]
    df = df.dropna(subset=features + ["Total cost"])
    X = df[features]
    y = df["Total cost"].astype(float)
    # Train/test split just for sanity (we re-fit on all data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # Build pipeline
    cat_feats = ["City", "Accommodation type", "Transportation type"]
    num_feats = ["Duration", "Year", "Month", "DayOfWeek", "IsWeekend", "Season"]
    pre = ColumnTransformer([
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_feats),
        ("num", "passthrough", num_feats)
    ])
    pipe = Pipeline([
        ("prep", pre),
        ("rf",   RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
    ])
    pipe.fit(X_train, y_train)
    # Optional in-app eval
    preds = pipe.predict(X_test)
    mae = np.mean(np.abs(preds - y_test))
    st.sidebar.write(f"🔍 Validation MAE: ${mae:,.2f}")
    # Save and return
    joblib.dump(pipe, model_path)
    return pipe

def main():
    st.title("💸 Travel Cost Estimator")
    ROOT       = Path(__file__).parent
    DATA_PATH  = ROOT / "data" / "Travel_details_dataset.csv"
    MODEL_PATH = ROOT / "models" / "travel_cost_rf.joblib"

    if not DATA_PATH.exists():
        st.error(f"❌ Data not found at {DATA_PATH}")
        st.stop()

    df = load_data(DATA_PATH)
    model = get_model(df, MODEL_PATH)

    # UI Inputs
    cities = sorted(df["City"].unique())
    accommodations = sorted(df["Accommodation type"].dropna().unique())
    transports = sorted(df["Transportation type"].dropna().unique())

    city    = st.selectbox("City", cities)
    start   = st.date_input("Start Date")
    end     = st.date_input("End Date", value=start)
    duration= (end - start).days or 1
    accom   = st.selectbox("Accommodation Type", accommodations)
    trans   = st.selectbox("Transportation Type", transports)

    if st.button("Estimate Cost"):
        inp = pd.DataFrame([{
            "City": city,
            "Accommodation type": accom,
            "Transportation type": trans,
            "Duration": duration,
            "Year": start.year,
            "Month": start.month,
            "DayOfWeek": start.weekday(),
            "IsWeekend": int(start.weekday() in [5,6]),
            "Season": ((start.month % 12)//3 + 1)
        }])
        cost = model.predict(inp)[0]
        st.metric("💰 Estimated Total Cost", f"${cost:,.2f}")

if __name__ == "__main__":
    main()
