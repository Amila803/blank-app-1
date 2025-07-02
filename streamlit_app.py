# streamlit_app.py

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
model_path = BASE_DIR / "model" / "Trip_cost_forecast_model.pkl"

st.title("💸 Travel Cost Predictor")

# Load model
if model_path.exists():
    model = joblib.load(model_path)
else:
    st.error("❌ Model file not found!")
    st.stop()

# Inputs
destination = st.selectbox("Select Destination", [
    "London, UK", "Phuket, Thailand", "Bali, Indonesia",
    "New York, USA", "Tokyo, Japan"
])
start_date = st.date_input("Trip Start Date")

# Predict
if st.button("Predict Cost"):
    month = start_date.month
    dayofweek = start_date.weekday()

    input_df = pd.DataFrame([{
        "Destination": destination,
        "month": month,
        "dayofweek": dayofweek
    }])

    prediction = model.predict(input_df)[0]
    st.success(f"Estimated Total Cost: ${prediction:,.2f}")
