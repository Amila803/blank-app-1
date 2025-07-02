import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
import joblib
import re

def clean_cost(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        # Remove dollar signs, commas, and any text
        cleaned = re.sub(r'[^\d.]', '', value)
        return float(cleaned) if cleaned else np.nan
    return float(value)

def load_and_preprocess_data(filepath):
    # Load data
    df = pd.read_csv(filepath)
    
    # Drop empty rows
    df = df.dropna(how='all')
    
    # Clean cost columns
    df['Accommodation cost'] = df['Accommodation cost'].apply(clean_cost)
    df['Transportation cost'] = df['Transportation cost'].apply(clean_cost)
    
    # Calculate total cost
    df['Total cost'] = df['Accommodation cost'] + df['Transportation cost']
    
    # Convert dates to datetime and extract features
    df['Start date'] = pd.to_datetime(df['Start date'], errors='coerce')
    df['End date'] = pd.to_datetime(df['End date'], errors='coerce')
    
    # Drop rows with invalid dates
    df = df.dropna(subset=['Start date', 'End date'])
    
    # Extract date features
    df['Year'] = df['Start date'].dt.year
    df['Month'] = df['Start date'].dt.month
    df['Season'] = df['Start date'].dt.month % 12 // 3 + 1
    df['Weekday'] = df['Start date'].dt.weekday
    
    # Clean destination names
    df['Destination'] = df['Destination'].str.split(',').str[0].str.strip()
    
    # Select relevant columns
    features = ['Destination', 'Duration (days)', 'Accommodation type', 
                'Transportation type', 'Year', 'Month', 'Season']
    target = 'Total cost'
    
    # Drop rows with missing values
    df = df.dropna(subset=features + [target])
    
    return df[features], df[target]

def train_model(X, y):
    # Define categorical and numerical features
    categorical_features = ['Destination', 'Accommodation type', 'Transportation type']
    numerical_features = ['Duration (days)', 'Year', 'Month', 'Season']
    
    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', 'passthrough', numerical_features)
        ])
    
    # Model pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Model MAE: ${mae:.2f}")
    
    return model

def main():
    # Load and preprocess data
    X, y = load_and_preprocess_data('Travel_details_dataset.csv')
    
    # Train model
    model = train_model(X, y)
    
    # Save model
    joblib.dump(model, 'travel_cost_model.joblib')
    print("Model saved as travel_cost_model.joblib")

if __name__ == '__main__':
    main()
