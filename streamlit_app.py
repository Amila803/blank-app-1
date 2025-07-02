import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import os
import sys
import re
import numpy as np
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Constants
MODEL_PATH = 'travel_cost_model.joblib'
DATA_PATH = 'Travel_details_dataset.csv'

DESTINATIONS = [
    'London', 'Phuket', 'Bali', 'New York', 'Tokyo', 'Paris', 'Sydney',
    'Rio de Janeiro', 'Amsterdam', 'Dubai', 'Cancun', 'Barcelona',
    'Honolulu', 'Berlin', 'Marrakech', 'Edinburgh', 'Rome', 'Bangkok',
    'Cape Town', 'Vancouver', 'Seoul', 'Los Angeles', 'Santorini',
    'Phnom Penh', 'Athens', 'Auckland'
]

ACCOMMODATION_TYPES = [
    'Hotel', 'Resort', 'Villa', 'Airbnb', 'Hostel', 'Riad',
    'Guesthouse', 'Vacation rental'
]

TRANSPORTATION_TYPES = [
    'Flight', 'Train', 'Plane', 'Bus', 'Car rental', 'Subway',
    'Ferry', 'Car'
]

def clean_cost(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        cleaned = re.sub(r'[^\d.]', '', value)
        return float(cleaned) if cleaned else np.nan
    return float(value)

def load_and_preprocess_data():
    try:
        # Check if file exists
        if not os.path.exists(DATA_PATH):
            st.error(f"Dataset file not found at: {os.path.abspath(DATA_PATH)}")
            st.info("Please make sure 'Travel_details_dataset.csv' is in the same directory as this script.")
            return None, None
            
        df = pd.read_csv(DATA_PATH)
        
        # Basic data validation
        if df.empty:
            st.warning("The dataset is empty!")
            return None, None
            
        # Data cleaning
        df = df.dropna(how='all')
        df['Accommodation cost'] = df['Accommodation cost'].apply(clean_cost)
        df['Transportation cost'] = df['Transportation cost'].apply(clean_cost)
        df['Total cost'] = df['Accommodation cost'] + df['Transportation cost']
        
        # Date handling
        df['Start date'] = pd.to_datetime(df['Start date'], errors='coerce')
        df['End date'] = pd.to_datetime(df['End date'], errors='coerce')
        df = df.dropna(subset=['Start date', 'End date'])
        
        # Feature engineering
        df['Year'] = df['Start date'].dt.year
        df['Month'] = df['Start date'].dt.month
        df['Season'] = df['Start date'].dt.month % 12 // 3 + 1
        df['Destination'] = df['Destination'].str.split(',').str[0].str.strip()
        
        # Additional feature engineering
        df['Duration (days)'] = (df['End date'] - df['Start date']).dt.days
        df['Is_peak_season'] = df['Month'].isin([6, 7, 8, 12]).astype(int)
        df['Is_long_trip'] = (df['Duration (days)'] > 14).astype(int)
        
        # Select features
        features = ['Destination', 'Duration (days)', 'Accommodation type', 
                   'Transportation type', 'Year', 'Month', 'Season',
                   'Is_peak_season', 'Is_long_trip']
        target = 'Total cost'
        
        # Final cleaning
        df = df.dropna(subset=features + [target])
        
        return df[features], df[target]
        
    except Exception as e:
        st.error(f"Error loading or processing data: {str(e)}")
        return None, None

def evaluate_model(model, X, y):
    """Evaluate model performance using cross-validation"""
    try:
        scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
        mae_scores = -scores
        st.write(f"Mean Absolute Error (CV): ${mae_scores.mean():.2f} (± {mae_scores.std():.2f})")
        
        scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        st.write(f"R² Score (CV): {scores.mean():.2f} (± {scores.std():.2f})")
    except Exception as e:
        st.warning(f"Could not perform cross-validation: {str(e)}")

def train_model():
    X, y = load_and_preprocess_data()
    
    if X is None or y is None:
        st.error("Cannot train model due to data issues.")
        return None
        
    categorical_features = ['Destination', 'Accommodation type', 'Transportation type']
    numerical_features = ['Duration (days)', 'Year', 'Month', 'Season', 'Is_peak_season', 'Is_long_trip']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', 'passthrough', numerical_features)
        ])
    
    # Define base models for stacking
    base_models = [
        ('random_forest', RandomForestRegressor(
            n_estimators=200, 
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )),
        ('svr', SVR(
            kernel='rbf',
            C=1.0,
            epsilon=0.1
        )),
        ('ridge', Ridge(
            alpha=1.0,
            random_state=42
        ))
    ]
    
    # Define meta-model
    meta_model = LinearRegression()
    
    # Create stacking ensemble
    stacked_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', StackingRegressor(
            estimators=base_models,
            final_estimator=meta_model,
            n_jobs=-1,
            passthrough=True  # Include original features along with base model predictions
        ))
    ])
    
    # Train model
    stacked_model.fit(X, y)
    
    # Evaluate model
    st.subheader("Model Evaluation")
    evaluate_model(stacked_model, X, y)
    
    try:
        joblib.dump(stacked_model, MODEL_PATH)
        st.success("Model trained and saved successfully!")
    except Exception as e:
        st.error(f"Error saving model: {str(e)}")
        
    return stacked_model

def load_model():
    if not os.path.exists(MODEL_PATH):
        st.warning("Model file not found. Training a new model...")
        return train_model()
        
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.warning("Attempting to train a new model...")
        return train_model()

def predict_cost(model, destination, start_date, duration, accommodation, transportation):
    if model is None:
        st.error("Cannot make predictions - no model available")
        return None
        
    try:
        # Calculate additional features
        month = start_date.month
        season = (month % 12) // 3 + 1
        is_peak_season = 1 if month in [6, 7, 8, 12] else 0
        is_long_trip = 1 if duration > 14 else 0
        
        input_data = pd.DataFrame({
            'Destination': [destination],
            'Duration (days)': [duration],
            'Accommodation type': [accommodation],
            'Transportation type': [transportation],
            'Year': [start_date.year],
            'Month': [month],
            'Season': [season],
            'Is_peak_season': [is_peak_season],
            'Is_long_trip': [is_long_trip]
        })
        
        prediction = model.predict(input_data)[0]
        return round(prediction, 2)
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def main():
    st.title('Travel Cost Estimator')
    st.write("Estimate the total cost of your trip based on destination, dates, and accommodation type.")
    
    # Sidebar for model management
    with st.sidebar:
        st.header("Model Management")
        if st.button("Retrain Model"):
            with st.spinner("Training new model..."):
                model = train_model()
        st.info("Click the button above to retrain the model with the latest data.")
    
    # Load or train model
    model = load_model()
    
    if model is None:
        st.error("Failed to initialize model. Cannot continue.")
        return
    
    with st.form("travel_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            destination = st.selectbox('Destination', DESTINATIONS)
            start_date = st.date_input('Start Date', min_value=datetime.today())
            duration = st.number_input('Duration (days)', min_value=1, max_value=90, value=7)
        
        with col2:
            accommodation = st.selectbox('Accommodation Type', ACCOMMODATION_TYPES)
            transportation = st.selectbox('Transportation Type', TRANSPORTATION_TYPES)
        
        submitted = st.form_submit_button("Estimate Cost")
    
    if submitted:
        total_cost = predict_cost(
            model, destination, start_date, duration, accommodation, transportation
        )
        
        if total_cost is not None:
            st.subheader("Estimated Cost")
            st.metric(label="Total Estimated Cost", value=f"${total_cost:,.2f}")
            
            st.write("**Estimated Cost Breakdown:**")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Accommodation", value=f"${total_cost * 0.7:,.2f}")
            with col2:
                st.metric(label="Transportation", value=f"${total_cost * 0.3:,.2f}")

if __name__ == '__main__':
    main()
