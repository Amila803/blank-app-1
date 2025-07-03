import pandas as pd
import pickle
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import re

def clean_cost(value):
    """Clean cost values that might contain currency symbols, commas, or text"""
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        cleaned = re.sub(r'[^\d.]', '', value.split('USD')[0].strip())
        return float(cleaned) if cleaned else np.nan
    return float(value)

def clean_destination(dest):
    """Clean destination names by extracting the main location"""
    if pd.isna(dest):
        return np.nan
    dest = str(dest).split(',')[0].strip()
    if dest == 'New York City':
        return 'New York'
    elif dest == 'Sydney, Aus' or dest == 'Sydney, AUS':
        return 'Sydney'
    elif dest == 'Bangkok, Thai':
        return 'Bangkok'
    elif dest == 'Phuket, Thai':
        return 'Phuket'
    elif dest == 'Cape Town, SA':
        return 'Cape Town'
    return dest

def load_and_preprocess_data():
    """Load and preprocess the dataset"""
    DATA_PATH = Path(__file__).parent / "data" / "Travel_details_dataset.csv"
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
    
    # Data cleaning
    df = df.dropna(how='all')
    df['Accommodation cost'] = df['Accommodation cost'].apply(clean_cost)
    df['Transportation cost'] = df['Transportation cost'].apply(clean_cost)
    df['Total cost'] = df['Accommodation cost'] + df['Transportation cost']
    df['Destination'] = df['Destination'].apply(clean_destination)
    df['Traveler nationality'] = df['Traveler nationality'].str.split().str[0].str.strip()
    df['Accommodation type'] = df['Accommodation type'].str.strip()
    df['Transportation type'] = df['Transportation type'].str.strip().replace({
        'Plane': 'Flight', 'Airplane': 'Flight'
    })
    
    # Date handling and feature engineering
    df['Start date'] = pd.to_datetime(df['Start date'], errors='coerce')
    df['End date'] = pd.to_datetime(df['End date'], errors='coerce')
    df = df.dropna(subset=['Start date', 'End date'])
    df['Year'] = df['Start date'].dt.year
    df['Month'] = df['Start date'].dt.month
    df['Season'] = df['Start date'].dt.month % 12 // 3 + 1
    df['Duration (days)'] = (df['End date'] - df['Start date']).dt.days
    df['Is_peak_season'] = df['Month'].isin([6, 7, 8, 12]).astype(int)
    df['Is_long_trip'] = (df['Duration (days)'] > 14).astype(int)
    df['Is_domestic'] = (df['Traveler nationality'] == df['Destination']).astype(int)
    
    # Final cleaning
    features = ['Destination', 'Traveler nationality', 'Duration (days)', 
               'Accommodation type', 'Transportation type', 'Year', 
               'Month', 'Season', 'Is_peak_season', 'Is_long_trip', 'Is_domestic']
    target = 'Total cost'
    
    df = df.dropna(subset=features + [target])
    df = df[(df['Total cost'] > 0) & (df['Total cost'] < 50000)]
    df = df[(df['Duration (days)'] > 0) & (df['Duration (days)'] <= 90)]
    
    return df[features], df[target]

def create_and_save_model():
    """Create the model and save it as my_file.pkl"""
    X, y = load_and_preprocess_data()
    
    categorical_features = ['Destination', 'Traveler nationality', 'Accommodation type', 'Transportation type']
    numerical_features = ['Duration (days)', 'Year', 'Month', 'Season', 'Is_peak_season', 'Is_long_trip', 'Is_domestic']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', 'passthrough', numerical_features)
        ])
    
    base_models = [
        ('random_forest', RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_split=5, random_state=42, n_jobs=-1)),
        ('svr', SVR(kernel='rbf', C=1.0, epsilon=0.1)),
        ('ridge', Ridge(alpha=1.0, random_state=42))
    ]
    
    stacked_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', StackingRegressor(
            estimators=base_models,
            final_estimator=LinearRegression(),
            n_jobs=-1,
            passthrough=True
        ))
    ])
    
    stacked_model.fit(X, y)
    
    # Save the model
    with open('my_file.pkl', 'wb') as f:
        pickle.dump(stacked_model, f)
    
    print("Model trained and saved as my_file.pkl")

if __name__ == '__main__':
    create_and_save_model()
