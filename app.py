import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("diabetes_model.pkl")

def make_prediction(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
    p = int(Pregnancies)
    g = float(Glucose)
    bp = float(BloodPressure)
    st = float(SkinThickness)
    ins = float(Insulin)
    bmi = float(BMI)
    dpf = float(DiabetesPedigreeFunction)
    age = int(Age)
    data = np.array([[p, g, bp, st, ins, bmi, dpf, age]])
    result = model.predict(data)[0]
    return result


# ------------ Streamlit UI ------------ #

st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺")

st.title("🩺 Diabetes Prediction App")
st.write("Fill the form below to check the diabetes risk:")

col1, col2 = st.columns(2)

with col1:
    Pregnancies = st.number_input("Number of Pregnancies", 0, 20, 1)
    Glucose = st.number_input("Glucose Level", 0.0, 300.0, 120.0)
    BloodPressure = st.number_input("Blood Pressure", 0.0, 200.0, 70.0)
    SkinThickness = st.number_input("Skin Thickness", 0.0, 100.0, 20.0)

with col2:
    Insulin = st.number_input("Insulin Level", 0.0, 900.0, 80.0)
    BMI = st.number_input("BMI", 0.0, 70.0, 25.0)
    DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", 0.0, 5.0, 0.5)
    Age = st.number_input("Age", 1, 120, 30)

if st.button("Predict"):
    result = make_prediction(Pregnancies, Glucose, BloodPressure, SkinThickness,
                             Insulin, BMI, DiabetesPedigreeFunction, Age)

    if result == 1:
        st.error("⚠️ High Risk of Diabetes! Please consult a doctor.")
    else:
        st.success("✅ You are likely Safe. No Diabetes Detected!")
