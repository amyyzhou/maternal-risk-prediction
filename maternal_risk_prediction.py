import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
df = pd.read_csv("Maternal Health Risk Data Set.csv")

# Display basic info
print(df.info())
print(df.head())

# Encode categorical target variable
label_encoder = LabelEncoder()
risk_mapping = {"low risk": 0, "mid risk": 1, "high risk": 2}
df["RiskLevel_encoded"] = df["RiskLevel"].map(risk_mapping)
# print(df[["RiskLevel", "RiskLevel_encoded"]].drop_duplicates()) -> verify correct labelling

# Define features and target
X = df.drop(columns=["RiskLevel", "RiskLevel_encoded"])
y = df["RiskLevel_encoded"]  # Use encoded risk level as the target

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(X_scaled)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)


# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Model Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Classification Report
# Classification Report
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=["low risk", "mid risk", "high risk"]))
# Precision = How many of the predicted positives were actually correct?
# Recall = How many of the actual positives were correctly predicted?

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot Confusion Matrix
plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
            xticklabels=["low risk", "mid risk", "high risk"], 
            yticklabels=["low risk", "mid risk", "high risk"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Get feature importance
feature_importances = model.feature_importances_
feature_names = X.columns

# Sort features by importance
sorted_idx = np.argsort(feature_importances)[::-1]

# Plot feature importance
plt.figure(figsize=(10,5))
sns.barplot(x=feature_importances[sorted_idx], y=np.array(feature_names)[sorted_idx], palette="coolwarm")
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Feature Importance in Predicting Maternal Health Risk")
plt.show()

import pickle

# Save trained model
with open("maternal_risk_model.pkl", "wb") as file:
    pickle.dump(model, file)

# Save scaler
with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)

print(f"Features used in training: {X.columns.tolist()}")
