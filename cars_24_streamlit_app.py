import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import pickle

st.title("Cars24 Used Car Price Prediction")

# numeric_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='median')),
#     ('scaler', StandardScaler())
# ])
# categorical_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='most_frequent')),
#     ('onehot', OneHotEncoder(handle_unknown='ignore'))
# ])

model = joblib.load("car_price_model.joblib")

year  = st.slider('Year of Manufacture', min_value=1995, max_value=2025, value=2015, step=1)

km_driven = st.number_input('Kilometers Driven', min_value=0, max_value=500000, value=50000, step=1000)

col1, col2, col3, col4 = st.columns(4)

with col1:
    fuel = st.selectbox('Fuel Type', options=['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'])

with col2:
    seller_type = st.selectbox('Seller Type', options=['Individual', 'Dealer', 'Trustmark Dealer'])

with col3:
    transmission = st.selectbox('Transmission Type', options=['Manual', 'Automatic'])

with col4:
    owner = st.selectbox('Owner Type', options=['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'])



mileage_km_ltr_kg = st.number_input('Mileage (km/ltr/kg)', min_value=0.0, max_value=50.0, value=15.0, step=0.1)

engine = st.number_input('Engine Capacity (CC)', min_value=500, max_value=5000, value=1500, step=50)

max_power = st.number_input('Max Power (bhp)', min_value=20.0, max_value=400.0, value=100.0, step=1.0)

seats = st.number_input('Number of Seats', min_value=2, max_value=10, value=5, step=1)

# Prediction button
if st.button("Predict Price"):
    input_data = pd.DataFrame({
        'year': [year],
        'km_driven': [km_driven],
        'fuel': [fuel],
        'seller_type': [seller_type],
        'transmission': [transmission],
        'owner': [owner],
        'mileage(km/ltr/kg)': [mileage_km_ltr_kg],
        'engine': [engine],
        'max_power': [max_power],
        'seats': [seats]
    })

    prediction = model.predict(input_data)
    st.success(f"The predicted price of the car is: ₹ {prediction[0]:,.2f}") 

