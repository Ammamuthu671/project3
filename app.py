import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="Heart Disease Prediction App",
    page_icon="❤️",
    layout="centered"
)

# Load the trained model
@st.cache_resource
def load_model():
    with open("heart_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# App Title and Description
st.title("❤️ Heart Disease Prediction App")
st.write("""
This application uses a Logistic Regression model to predict the likelihood of heart disease 
based on clinical parameters. Please input the patient's data below.
""")

st.markdown("---")

# Creating form layout using columns
st.subheader("Patient Clinical Data")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50, step=1)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female (0)" if x == 0 else "Male (1)")
    cp = st.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3], 
                      format_func=lambda x: f"Type {x}")
    trestbps = st.number_input("Resting Blood Pressure (trestbps in mm Hg)", min_value=50, max_value=250, value=120)
    chol = st.number_input("Serum Cholestoral (chol in mg/dl)", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", options=[0, 1], format_func=lambda x: "False (0)" if x == 0 else "True (1)")
    restecg = st.selectbox("Resting Electrocardiographic Results (restecg)", options=[0, 1, 2])

with col2:
    thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise Induced Angina (exang)", options=[0, 1], format_func=lambda x: "No (0)" if x == 1 else "Yes (1)")
    oldpeak = st.number_input("ST Depression Induced by Exercise (oldpeak)", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
    slope = st.selectbox("Slope of the Peak Exercise ST Segment (slope)", options=[0, 1, 2])
    ca = st.selectbox("Number of Major Vessels Colored by Flourosopy (ca)", options=[0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia (thal)", options=[0, 1, 2, 3])

st.markdown("---")

# Prediction logic
if st.button("Predict Heart Disease", type="primary"):
    # Arrange inputs exactly in the order the model expects (13 features)
    input_data = np.array([[
        age, sex, cp, trestbps, chol, fbs, restecg, 
        thalach, exang, oldpeak, slope, ca, thal
    ]])
    
    # Get prediction and probabilities
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    
    # Display Results
    st.subheader("Prediction Result")
    
    if prediction == 1:
        st.error(f"🚨 **High Risk:** The model predicts a high probability of heart disease.")
        st.write(f"Confidence: **{probabilities[1] * 100:.2f}%**")
    else:
        st.success(f"✅ **Low Risk:** The model predicts a low probability of heart disease.")
        st.write(f"Confidence: **{probabilities[0] * 100:.2f}%**")

    # Optional: Display metrics breakdown
    st.progress(float(probabilities[1]))
