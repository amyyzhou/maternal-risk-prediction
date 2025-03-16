# 🩺 **Maternal Health Risk Prediction** 🤰

## **📖 Overview**
**Early assessment of maternal health risk can prevent complications during pregnancy.**
This project uses machine learning to predict maternal risk levels based on six key physiological features:
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
