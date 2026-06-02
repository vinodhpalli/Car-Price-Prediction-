import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load('model.pkl')

# Title
st.title("KNN Regression App")

st.write("Enter values")

# Inputs
feature1 = st.number_input("Feature 1")
feature2 = st.number_input("Feature 2")
feature3 = st.number_input("Feature 3")
feature4 = st.number_input("Feature 4")

# Predict button
if st.button("Predict"):

    features = np.array([
        [feature1, feature2, feature3, feature4]
    ])

    prediction = model.predict(features)

    st.success(f"Prediction: {prediction[0]}")