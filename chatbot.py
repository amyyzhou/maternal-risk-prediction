import streamlit as st
import requests

st.title("🤰 Maternal Health Risk Chatbot")
st.write("Enter your details to assess your maternal health risk.")

# User input fields
age = st.number_input("Age", min_value=15, max_value=50, step=1)
systolic_bp = st.number_input("Systolic Blood Pressure", min_value=80, max_value=200, step=1)
diastolic_bp = st.number_input("Diastolic Blood Pressure", min_value=50, max_value=120, step=1)
blood_sugar = st.number_input("Blood Sugar Level (BS)", min_value=2.0, max_value=20.0, step=0.1)
body_temp = st.number_input("Body Temperature (°F)", min_value=95.0, max_value=105.0, step=0.1)
heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=150, step=1)

# Submit button
if st.button("Predict Risk"):
    data = {
        "Age": age,
        "SystolicBP": systolic_bp,
        "DiastolicBP": diastolic_bp,
        "BS": blood_sugar,
        "BodyTemp": body_temp,
        "HeartRate": heart_rate
    }

    # Send request to Flask API
    API_URL = "https://maternal-risk-prediction.onrender.com/predict"
    result = API_URL.json()

    # Display predicted result
    if "Risk Level" in result:
        st.success(f"🩺 Predicted Maternal Health Risk: **{result['Risk Level']}**")
    else:
        st.error(f"Error: {result['error']}")
