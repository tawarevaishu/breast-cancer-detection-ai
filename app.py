import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ------------------------------
# Load and train model
# ------------------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="🩺 Breast Cancer Detection", page_icon="💖", layout="wide")
st.title("🩷 Breast Cancer Detection using AI/ML")
st.write("Enter the feature values below and click **Predict** to check the diagnosis.")

# Input fields
feature_values = []
cols = st.columns(3)

for i, feature in enumerate(data.feature_names):
    with cols[i % 3]:
        value = st.number_input(feature, float(X[feature].min()), float(X[feature].max()), float(X[feature].mean()))
        feature_values.append(value)

# Predict button
if st.button("🔍 Predict"):
    features_array = np.array(feature_values).reshape(1, -1)
    features_scaled = scaler.transform(features_array)
    prediction = model.predict(features_scaled)[0]
    result = "🩸 Malignant (Cancer Detected)" if prediction == 0 else "💚 Benign (No Cancer Detected)"

    # Display result
    if prediction == 0:
        st.error(result)
    else:
        st.success(result)

# Footer
st.markdown("---")
st.caption("Developed by Vaishnavi Taware 💻 | Powered by Streamlit & scikit-learn")
