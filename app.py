import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load trained model and scaler
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')  # Save and load the scaler for consistent preprocessing

# Streamlit app title
st.title("UPI Transaction Fraud Detection")

# User input form
with st.form("fraud_detection_form"):
    # Collect user inputs
    name = st.text_input("Name", placeholder="Enter your name")
    transaction_id = st.text_input("Transaction ID", placeholder="Enter transaction ID")
    phone_number = st.text_input("Phone Number", placeholder="Enter phone number")
    state = st.text_input("State", placeholder="Enter state of transaction")
    step = st.number_input("Step (Time)", min_value=1, value=1, step=1, help="Represents the time step (hour).")
    transaction_type = st.selectbox("Transaction Type", options=["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"])
    amount = st.number_input("Transaction Amount", min_value=0.0, value=0.0, step=0.01)
    old_balance_orig = st.number_input("Old Balance Origin", min_value=0.0, value=0.0, step=0.01)
    new_balance_orig = st.number_input("New Balance Origin", min_value=0.0, value=0.0, step=0.01)
    old_balance_dest = st.number_input("Old Balance Destination", min_value=0.0, value=0.0, step=0.01)
    new_balance_dest = st.number_input("New Balance Destination", min_value=0.0, value=0.0, step=0.01)

    # Submit button
    submitted = st.form_submit_button("Check for Fraud")

# Handle prediction after form submission
if submitted:
    # Encode transaction type
    type_mapping = {"PAYMENT": 0, "TRANSFER": 1, "CASH_OUT": 2, "DEBIT": 3, "CASH_IN": 4}
    transaction_type_encoded = type_mapping.get(transaction_type, 0)

    # Create the feature array
    transaction_data = np.array([[step, transaction_type_encoded, amount, old_balance_orig, new_balance_orig, old_balance_dest, new_balance_dest]])

    # Scale the input data
    transaction_data_scaled = scaler.transform(transaction_data)

    # Perform prediction
    prediction = model.predict(transaction_data_scaled)

    # Display result
    st.header("Prediction Result")
    if prediction[0] == 1:
        st.error("🚨 The transaction is predicted to be **fraudulent**!")
    else:
        st.success("✅ The transaction is predicted to be **legitimate**.")
