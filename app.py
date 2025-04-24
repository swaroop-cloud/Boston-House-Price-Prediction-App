import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('house_price_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title('🏠 House Price Prediction App')

# Input fields for user
CRIM = st.number_input('Crime Rate (CRIM)', 0.0, 100.0, 0.1)
ZN = st.number_input('Residential Land (ZN)', 0.0, 100.0, 0.0)
INDUS = st.number_input('Industrial Area (INDUS)', 0.0, 30.0, 0.0)
CHAS = st.selectbox('Charles River Dummy (CHAS)', [0, 1])
NOX = st.number_input('Nitric Oxide (NOX)', 0.0, 1.0, 0.1)
RM = st.number_input('Avg Rooms (RM)', 1.0, 10.0, 5.0)
AGE = st.number_input('Age of Building (AGE)', 0.0, 100.0, 50.0)
DIS = st.number_input('Distance to Employment (DIS)', 0.0, 12.0, 4.0)
RAD = st.number_input('Accessibility Index (RAD)', 1, 24, 1)
TAX = st.number_input('Property Tax (TAX)', 100.0, 800.0, 300.0)
PTRATIO = st.number_input('Pupil-Teacher Ratio (PTRATIO)', 10.0, 30.0, 15.0)
B = st.number_input('Proportion of Black Population (B)', 0.0, 400.0, 300.0)
LSTAT = st.number_input('% Lower Status (LSTAT)', 0.0, 40.0, 5.0)

# Predict button
if st.button('Predict Price'):
    features = np.array([[CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]])
    features_scaled = scaler.transform(features)
    price = model.predict(features_scaled)
    st.success(f'Predicted House Price: ${price[0]*1000:.2f}')
