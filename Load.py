import streamlit as st
import numpy as np
import pandas as pd
from joblib import load
import os
import matplotlib.pyplot as plt

# Paths for model and scaler files
model_file = "ANN_best_model(Load (kN)).joblib"
feature_scaler_file = "feature_scaler.pkl"
target_scaler_file = "target_scaler.pkl"

# Load the trained ANN model and scalers
try:
    ann_model = load(model_file)
    feature_scaler = load(feature_scaler_file)
    target_scaler = load(target_scaler_file)
except Exception as e:
    st.error(f"Error loading model or scalers: {e}")

# Feature names
feature_names = [
    "Diameter (D) (mm):",
    "Length (L) (mm):",
    "Thickness (t) (mm):",
    "Edge distance (e) (mm):",
    "Proof strength (f0.2) (MPa):",
    "Ultimate strength (fu) (MPa):",
    "Strain hardening parameter (n):"
]

# Title and Description
st.title("Buckling Resistance Prediction Using ANN")
st.markdown(
    "This app predicts the buckling resistance of stainless steel CHS beam-columns based on user input parameters "
    "using an Artificial Neural Network (ANN) model."
)

# Sidebar Inputs
st.sidebar.header("Input Features")
inputs = {}
for feature in feature_names:
    label = feature.split(" (")[0]
    inputs[label] = st.sidebar.number_input(f"{feature}", value=1.0, step=0.1)

# Scale features
def scale_features(feature_values):
    try:
        scaled_values = feature_scaler.transform([feature_values])
        return scaled_values
    except Exception as e:
        st.error(f"Error scaling features: {e}")
        return None

# Prediction
if st.sidebar.button("Predict"):
    if ann_model is None:
        st.error("Model not loaded. Please check your setup.")
    else:
        feature_values = [inputs[name.split(" (")[0]] for name in feature_names]
        scaled_input = scale_features(feature_values)
        if scaled_input is not None:
            scaled_result = ann_model.predict(scaled_input)[0]
            result = target_scaler.inverse_transform([[scaled_result]])[0][0]
            st.subheader("Predicted Output")
            st.write(f"**Buckling Resistance  (Nu):** {result:.2f} kN")

# Prediction History
if "history" not in st.session_state:
    st.session_state.history = []

if st.sidebar.button("Save to History"):
    if ann_model:
        feature_values = [inputs[name.split(" (")[0]] for name in feature_names]
        scaled_input = scale_features(feature_values)
        if scaled_input is not None:
            scaled_result = ann_model.predict(scaled_input)[0]
            result = target_scaler.inverse_transform([[scaled_result]])[0][0]
            st.session_state.history.append({
                "features": inputs,
                "Load (kN)": result
            })
        st.success("Prediction saved to history.")

if st.session_state.history:
    st.subheader("Prediction History")
    history_df = pd.DataFrame(st.session_state.history)
    st.write(history_df)

# Plot History
if st.session_state.history:
    if st.sidebar.button("Plot History"):
        loads = [record['Load (kN)'] for record in st.session_state.history]
        plt.figure(figsize=(6, 4))
        plt.plot(loads, marker='o', linestyle='-', color='b', label='Load (kN)')
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
