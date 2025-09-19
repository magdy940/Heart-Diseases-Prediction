import streamlit as st
import pickle
import pandas as pd

# Load the trained pipeline
with open("heart_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

st.title("❤️ Heart Disease Prediction App")
st.write("This app predicts the likelihood of heart disease based on clinical parameters.")

# Input fields
age = st.number_input("Age", min_value=1, max_value=120, value=40)
sex = st.selectbox("Sex", ["M", "F"])
cp = st.selectbox("Chest Pain Type", ["ASY", "ATA", "NAP", "TA"])
resting_bp = st.number_input("Resting Blood Pressure", min_value=50, max_value=250, value=120)
cholesterol = st.number_input("Cholesterol", min_value=0, max_value=600, value=200)
fasting_bs = st.selectbox("Fasting Blood Sugar ( >120 mg/dl )", [0, 1])
rest_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.number_input("Maximum Heart Rate", min_value=60, max_value=220, value=150)
exercise_angina = st.selectbox("Exercise Induced Angina", ["Y", "N"])
oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# Create input dataframe
input_data = pd.DataFrame({
    "Age": [age],
    "Sex": [sex],
    "ChestPainType": [cp],
    "RestingBP": [resting_bp],
    "Cholesterol": [cholesterol],
    "FastingBS": [fasting_bs],
    "RestingECG": [rest_ecg],
    "MaxHR": [max_hr],
    "ExerciseAngina": [exercise_angina],
    "Oldpeak": [oldpeak],
    "ST_Slope": [st_slope]
})

# Prediction
if st.button("Predict"):
    prediction = pipeline.predict(input_data)[0]
    prediction_proba = pipeline.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ The patient is likely to have heart disease.\n\nProbability: {prediction_proba:.2f}")
    else:
        st.success(f"✅ The patient is likely healthy.\n\nProbability: {prediction_proba:.2f}")

