# 🩺 **Maternal Health Risk Prediction** 🤰

## **📖 Overview**
**Early assessment of maternal health risk can prevent complications during pregnancy.**

📂 Maternal-Health-Risk-Prediction
```
│── 📄 app.py                # Flask API for model prediction
│── 📄 chatbot.py            # Streamlit chatbot UI
│── 📄 maternal_risk_prediction.py  # Model training and evaluation
│── 📄 requirements.txt      # Required dependencies
│── 📄 Maternal Health Risk Data Set.csv  # Dataset
│── 📄 maternal_risk_model.pkl  # Trained model file
│── 📄 scaler.pkl            # StandardScaler object for input transformation
└── 📄 README.md             # Project documentation
```

## **🛠 Installation & Setup**
Ensure you have Python 3.7+ installed. Then, install the dependencies:
```pip install -r requirements.txt```

## **⚡ Running the Project**
### **Step 1: Train the Model**
Run the training script to generate the trained model (```maternal_risk_model.pkl```) and scaler (```scaler.pkl```):

```python maternal_risk_prediction.py```

### **Step 2: Start the Flask API**
Run the Flask server to handle prediction requests:
```python app.py```

### **Step 3: Start the Chatbot UI**
In a new terminal, run:
```streamlit run chatbot.py```

## **1️⃣ Machine Learning Model: Random Forest**
The machine learning model is responsible for predicting maternal health risk levels based on six key physiological factors:
- Age
- Systolic Blood Pressure
- Diastolic Blood Pressure
- Blood Sugar Levels (BS)
- Body Temperature (°F)
- Heart Rate (bpm)

The model categorizes maternal health risk into:
- Low Risk (0)
- Mid Risk (1)
- High Risk (2)

1️⃣ Data Preprocessing:
- The dataset (```Maternal Health Risk Data Set.csv```) is loaded using pandas.
- Categorical labels (Low Risk, Mid Risk, High Risk) are encoded as numbers (0, 1, 2).
- Features are standardized using ```StandardScaler()``` for better model performance.

2️⃣ Model Training:
- The data is split into training (80%) and testing (20%) sets.
- A Random Forest Classifier is trained on the data.
- The trained model is evaluated using: Accuracy Score, Confusion Matrix, Feature Importance Analysis

3️⃣ Model Saving:
- The trained Random Forest model is saved as ```maternal_risk_model.pkl```.
- The scaler is saved as ```scaler.pkl``` for consistent input transformation.

## **2️⃣ Flask API: Backend Server**
The Flask API serves as the backend system that takes user data, processes it, and returns a risk prediction.

1️⃣ Receives HTTP Requests:
- The API listens for ```POST``` requests at ```/predict```.
- Each request contains a JSON object with patient health parameters.

2️⃣ Processes Input:
- Checks if all required features (Age, BP, BS, etc.) are provided.
- Converts data into a structured numpy array.
- Applies the trained scaler (```scaler.pkl```) to standardize the input.

3️⃣ Generates Prediction:
- The trained model (```maternal_risk_model.pkl```) makes a prediction.
- The numeric output (0, 1, 2) is mapped back to ```"Low Risk"```, ```"Mid Risk"```, or ```"High Risk"```.

4️⃣ Returns JSON Response

## **3️⃣ Streamlit Chatbot: User Interface**
The Streamlit chatbot provides an easy-to-use web interface where users can enter their health data and receive an instant risk assessment.

1️⃣ User Input:
The chatbot asks users to enter:
- Age
- Systolic Blood Pressure
- Diastolic Blood Pressure
- Blood Sugar Levels (BS)
- Body Temperature (°F)
- Heart Rate (bpm)
The user clicks "Predict Risk".

2️⃣ Sends Data to Flask API:
- The chatbot packages user input into JSON.
- Sends a POST request to the Flask API at http://127.0.0.1:5000/predict.

3️⃣ Receives & Displays Prediction:
- The API responds with the predicted risk level.
- The chatbot displays the result in a user-friendly format.
