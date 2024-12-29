import streamlit as st
import numpy as np
import pandas as pd
from joblib import load
from sklearn.preprocessing import MinMaxScaler
import os
import matplotlib.pyplot as plt

# Paths for model and Excel file
model_file = "ANN_best_model(Load (kN)).joblib"
excel_file = "14.xlsx"

# Load the trained ANN model
try:
    gb_model = load(model_file)
except Exception as e:
    gb_model = None
    st.error(f"Failed to load the model: {e}")

# Feature names
feature_names = [
    "The outer diameter, D (mm):",
    "The length , L (mm):",
    "The thickness , t (mm):",
    "The eccentricity, e (mm):",
    "The yield strength at 0.2 % offset , f0.2 (MPa):",
    "The ultimate strength  , fu (MPa):",
    "The Romberg-Osgood exponent, n:",
]

# Title and Description
st.title("Buckling Resistance Prediction")
st.markdown("This app predicts the buckling resistance of CHS SS members.")

# Sidebar Inputs
st.sidebar.header("Input Features")
inputs = {}
for feature in feature_names:
    label = feature.split(",")[0]
    inputs[label] = st.sidebar.number_input(f"{feature}", value=1.0, step=0.1)

# Scale features
def scale_features(feature_values):
    try:
        df = pd.read_excel(excel_file, names=[name.split(",")[0] for name in feature_names] + ["Load (kN)"])
        X = df.drop(columns=["Load (kN)"])
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(X)
        return scaler.transform([feature_values])
    except Exception as e:
        st.error(f"Error scaling features: {e}")
        return None

# Prediction
if st.sidebar.button("Predict"):
    if gb_model is None:
        st.error("Model not loaded. Please check your setup.")
    else:
        feature_values = [inputs[name.split(",")[0]] for name in feature_names]
        scaled_input = scale_features(feature_values)
        if scaled_input is not None:
            result = gb_model.predict(scaled_input)[0]
            st.subheader("Predicted Output")
            st.write(f"**The buckling resistance  (Nu):** {result:.2f} kN")

# Prediction History
if "history" not in st.session_state:
    st.session_state.history = []

if st.sidebar.button("Save to History"):
    if gb_model:
        st.session_state.history.append({
            "features": inputs,
            "Load (kN)": gb_model.predict(scale_features([inputs[name.split(',')[0]] for name in feature_names]))[0]
        })
    st.success("Prediction saved to history.")

if st.session_state.history:
    st.subheader("Prediction History")
    for i, record in enumerate(st.session_state.history, start=1):
        st.write(f"**Prediction {i}:** Features: {record['features']}, Load (kN): {record['Load (kN)']:.2f} kN")

# Plot History
if st.session_state.history:
    if st.sidebar.button("Plot History"):
        results = [record['Load (kN)'] for record in st.session_state.history]
        plt.figure(figsize=(6, 4))
        plt.plot(results, marker='o', linestyle='-', color='b', label='Load (kN)')
        plt.title("Prediction History")
        plt.xlabel("Prediction Count")
        plt.ylabel("Load (kN)")
        plt.grid()
        plt.legend()
        st.pyplot(plt)

# Save Results
if st.sidebar.button("Save Results to CSV"):
    try:
        csv_path = 'prediction_results.csv'
        pd.DataFrame(st.session_state.history).to_csv(csv_path, index=False)
        st.success(f"Results saved to {csv_path}")
    except Exception as e:
        st.error(f"Error saving results: {e}")

# Footer
st.markdown("---")
st.info("Developed by Sina Sarfarazi, University of Naples Federico II, Italy.\nContact: sina.srfz@gmail.com")
