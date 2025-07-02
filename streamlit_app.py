import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# Load the trained model
model = joblib.load('travel_cost_model.joblib')

# Destination options (extracted from dataset)
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

def predict_cost(destination, start_date, duration, accommodation, transportation):
    # Prepare input data
    input_data = pd.DataFrame({
        'Destination': [destination],
        'Duration (days)': [duration],
        'Accommodation type': [accommodation],
        'Transportation type': [transportation],
        'Year': [start_date.year],
        'Month': [start_date.month],
        'Season': [(start_date.month % 12) // 3 + 1]
    })
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    return round(prediction, 2)

def main():
    st.title('Travel Cost Estimator')
    st.write("""
    Estimate the total cost of your trip based on destination, dates, and accommodation type.
    """)
    
    # Input form
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
        # Make prediction
        total_cost = predict_cost(
            destination, start_date, duration, accommodation, transportation
        )
        
        # Display results
        st.subheader("Estimated Cost")
        st.metric(label="Total Estimated Cost", value=f"${total_cost:,.2f}")
        
        # Breakdown (approximate based on model features)
        st.write("**Estimated Cost Breakdown:**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Accommodation", value=f"${total_cost * 0.7:,.2f}")
        with col2:
            st.metric(label="Transportation", value=f"${total_cost * 0.3:,.2f}")

if __name__ == '__main__':
    main()
