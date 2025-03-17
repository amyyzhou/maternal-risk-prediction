import streamlit as st
import requests

st.title("🤰 Maternal Health Risk Chatbot")
st.write("Enter your details to assess your maternal health risk.")

API_URL = "https://maternal-risk-prediction.onrender.com/predict"  # ✅ Correct URL

# User Input Fields
age = st.number_input("Age", min_value=15, max_value=50, step=1)
systolic_bp = st.number_input("Systolic Blood Pressure", min_value=80, max_value=200, step=1)
diastolic_bp = st.number_input("Diastolic Blood Pressure", min_value=50, max_value=120, step=1)
blood_sugar = st.number_input("Blood Sugar Level (BS)", min_value=2.0, max_value=20.0, step=0.1)
body_temp = st.number_input("Body Temperature (°F)", min_value=95.0, max_value=105.0, step=0.1)
heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=150, step=1)

if st.button("Predict Risk"):
    data = {
        "Age": age,
        "SystolicBP": systolic_bp,
        "DiastolicBP": diastolic_bp,
        "BS": blood_sugar,
        "BodyTemp": body_temp,
        "HeartRate": heart_rate
    }

    try:
        # Send request to Flask API
        response = requests.post(API_URL, json=data)

        # ✅ Correct: Now calling `.json()` on an actual HTTP response
        if response.status_code == 200:
            result = response.json()
            st.success(f"🩺 Predicted Maternal Health Risk: **{result['Risk Level']}**")
        else:
            st.error(f"Server Error: {response.text}")

    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")
