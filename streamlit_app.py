import streamlit as st
import pandas as pd
import pickle
from datetime import datetime
import os

# Constants
MODEL_PATH = 'my_file.pkl'

# Destination list
DESTINATIONS = [
    'London', 'Phuket', 'Bali', 'New York', 'Tokyo', 'Paris', 'Sydney',
    'Rio de Janeiro', 'Amsterdam', 'Dubai', 'Cancun', 'Barcelona',
    'Honolulu', 'Berlin', 'Marrakech', 'Edinburgh', 'Rome', 'Bangkok',
    'Cape Town', 'Vancouver', 'Seoul', 'Los Angeles', 'Santorini',
    'Phnom Penh', 'Athens', 'Auckland'
]

# Nationality list
NATIONALITIES = sorted(list(set([
    'American', 'Canadian', 'Korean', 'British', 'Vietnamese', 'Australian',
    'Brazilian', 'Dutch', 'Emirati', 'Mexican', 'Spanish', 'Chinese',
    'German', 'Moroccan', 'Scottish', 'Indian', 'Italian', 'South Korean',
    'Taiwanese', 'South African', 'French', 'Japanese', 'Cambodia', 'Greece',
    'United Arab Emirates', 'Hong Kong', 'Singapore', 'Indonesia', 'USA',
    'UK', 'China', 'New Zealander'
])))

ACCOMMODATION_TYPES = [
    'Hotel', 'Resort', 'Villa', 'Airbnb', 'Hostel', 'Riad',
    'Guesthouse', 'Vacation rental'
]

TRANSPORTATION_TYPES = [
    'Flight', 'Train', 'Plane', 'Bus', 'Car rental', 'Subway',
    'Ferry', 'Car', 'Airplane'
]

def load_model():
    """Load the pre-trained model from pickle file"""
    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found. Please run train_and_save_model.py first.")
        return None
        
    try:
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def predict_cost(model, nationality, destination, start_date, duration, accommodation, transportation):
    """Make predictions using the loaded model"""
    if model is None:
        st.error("Cannot make predictions - no model available")
        return None
        
    try:
        month = start_date.month
        season = (month % 12) // 3 + 1
        is_peak_season = 1 if month in [6, 7, 8, 12] else 0
        is_long_trip = 1 if duration > 14 else 0
        is_domestic = 1 if nationality == destination else 0
        
        transportation = 'Flight' if transportation in ['Plane', 'Airplane'] else transportation
        
        input_data = pd.DataFrame({
            'Destination': [destination],
            'Traveler nationality': [nationality],
            'Duration (days)': [duration],
            'Accommodation type': [accommodation],
            'Transportation type': [transportation],
            'Year': [start_date.year],
            'Month': [month],
            'Season': [season],
            'Is_peak_season': [is_peak_season],
            'Is_long_trip': [is_long_trip],
            'Is_domestic': [is_domestic]
        })
        
        prediction = model.predict(input_data)[0]
        return round(prediction, 2)
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def main():
    st.title('Travel Cost Estimator')
    st.write("Estimate the total cost of your trip based on your nationality, destination, dates, and accommodation type.")
    
    # Load model
    model = load_model()
    
    if model is None:
        st.stop()
    
    with st.form("travel_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            nationality = st.selectbox('Your Nationality', NATIONALITIES, index=0)
            destination = st.selectbox('Destination', DESTINATIONS)
            start_date = st.date_input('Start Date', min_value=datetime.today())
        
        with col2:
            duration = st.number_input('Duration (days)', min_value=1, max_value=90, value=7)
            accommodation = st.selectbox('Accommodation Type', ACCOMMODATION_TYPES)
            transportation = st.selectbox('Transportation Type', TRANSPORTATION_TYPES)
        
        submitted = st.form_submit_button("Estimate Cost")
    
    if submitted:
        total_cost = predict_cost(
            model, nationality, destination, start_date, duration, accommodation, transportation
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
            
            if nationality == destination:
                st.info("This is a domestic trip (same nationality and destination)")
            else:
                st.info("This is an international trip")

if __name__ == '__main__':
    main()
