import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("student_model.pkl")

# Load dataset structure
df_original = pd.read_csv("data/student-mat.csv", sep=";")
df_original = df_original.select_dtypes(include=["int64"])

st.title("Student Performance Predictor")

st.write("Enter student academic details")

# User inputs
g1 = st.number_input("G1 marks")
g2 = st.number_input("G2 marks")
absences = st.number_input("Absences")

# Create input row with default values
input_data = df_original.drop("G3", axis=1).iloc[0:1].copy()

input_data["G1"] = g1
input_data["G2"] = g2
input_data["absences"] = absences

if st.button("Predict Final Grade"):
    prediction = model.predict(input_data)
    st.success(f"Predicted Final Grade: {prediction[0]}")