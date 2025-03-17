from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)  # ✅ Allow cross-origin requests

# Load model and scaler
with open("maternal_risk_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Expected features
expected_features = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]

# Risk level mapping
risk_mapping = {0: "Low Risk", 1: "Mid Risk", 2: "High Risk"}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        missing_features = [feat for feat in expected_features if feat not in data]
        if missing_features:
            return jsonify({"error": f"Missing features: {', '.join(missing_features)}"}), 400

        features = np.array([data[feat] for feat in expected_features]).reshape(1, -1)
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        risk_level = risk_mapping[prediction]

        return jsonify({"Risk Level": risk_level})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
