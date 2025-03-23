from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS

# Load the trained model and scaler
with open("maternal_risk_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Expected features (MUST match the training dataset)
expected_features = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]

# Define risk level mapping
risk_mapping = {0: "Low Risk", 1: "Mid Risk", 2: "High Risk"}

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # Check if all required features are provided
        missing_features = [feat for feat in expected_features if feat not in data]
        if missing_features:
            return jsonify({"error": f"Missing features: {', '.join(missing_features)}"}), 400

        # Convert input data into numpy array
        features = np.array([data[feat] for feat in expected_features]).reshape(1, -1)

        # Standardize the input data
        features_scaled = scaler.transform(features)

        # Make a prediction
        prediction = model.predict(features_scaled)[0]
        risk_level = risk_mapping[prediction]

        return jsonify({"Risk Level": risk_level})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)