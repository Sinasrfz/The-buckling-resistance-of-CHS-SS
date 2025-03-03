import streamlit as st
import numpy as np
import pandas as pd
from joblib import load
import os
import matplotlib.pyplot as plt

# Input ranges based on training data
D_min, D_max = 40, 160 
L_min, L_max = 160, 1920 
t_min, t_max = 3, 9
e_min, e_max = 10, 100
f0_2_min, f0_2_max = 290, 519
fu_min, fu_max = 728, 799
n_min, n_max = 4.9, 7.8

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
    "Diameter (D) (mm)",
    "Length (L) (mm)",
    "Thickness (t) (mm)",
    "Edge distance (e) (mm)",
    "Proof strength (f0.2) (MPa)",
    "Ultimate strength (fu) (MPa)",
    "Strain hardening parameter (n)"
]

# Title and Description
st.title("Buckling Resistance Prediction Using ANN")

# Initialize session state for Read Me toggle
if "show_readme" not in st.session_state:
    st.session_state.show_readme = False

# Toggle Read Me display
if st.sidebar.button("Read Me"):
    st.session_state.show_readme = not st.session_state.show_readme

if st.session_state.show_readme:
    st.subheader("README for Buckling Resistance Prediction GUI")
    st.markdown("""
    **How to Use**:
    1. **Input Features**:
       - **D (mm)**: Diameter (40–160).
       - **L (mm)**: Length (160–1920).
       - **t (mm)**: Thickness (3–9).
       - **e (mm)**: Edge distance (10–100).
       - **f0.2 (MPa)**: Proof strength (290–519).
       - **fu (MPa)**: Ultimate strength (728–799).
       - **n**: Strain hardening parameter (4.9–7.8).
    2. **Validation**:
       - Ensure inputs are within valid ranges. Errors will be displayed if values are invalid.
    3. **Prediction**:
       - Click "Predict" after entering valid inputs to compute buckling resistance.
    4. **Prediction History**:
       - Save and view previous predictions.
    5. **Visualization**:
       - View the trend of past predictions in graphical format.

    For support, contact: **Sina Sarfarazi** (Email: sina.srfz@gmail.com).
    """)

# Sidebar Inputs
st.sidebar.header("Enter Input Features")
error = False

D = st.sidebar.number_input("Diameter (D) [mm]", min_value=D_min, max_value=D_max, step=1.0, value=50.0)
if D < D_min or D > D_max:
    st.sidebar.error(f"D must be between {D_min} and {D_max} mm.")
    error = True

L = st.sidebar.number_input("Length (L) [mm]", min_value=L_min, max_value=L_max, step=1.0, value=500.0)
if L < L_min or L > L_max:
    st.sidebar.error(f"L must be between {L_min} and {L_max} mm.")
    error = True

t = st.sidebar.number_input("Thickness (t) [mm]", min_value=t_min, max_value=t_max, step=0.1, value=5.0)
if t < t_min or t > t_max:
    st.sidebar.error(f"t must be between {t_min} and {t_max} mm.")
    error = True

e = st.sidebar.number_input("Edge distance (e) [mm]", min_value=e_min, max_value=e_max, step=1.0, value=30.0)
if e < e_min or e > e_max:
    st.sidebar.error(f"e must be between {e_min} and {e_max} mm.")
    error = True

f0_2 = st.sidebar.number_input("Proof strength (f0.2) [MPa]", min_value=f0_2_min, max_value=f0_2_max, step=1.0, value=350.0)
if f0_2 < f0_2_min or f0_2 > f0_2_max:
    st.sidebar.error(f"f0.2 must be between {f0_2_min} and {f0_2_max} MPa.")
    error = True

fu = st.sidebar.number_input("Ultimate strength (fu) [MPa]", min_value=fu_min, max_value=fu_max, step=1.0, value=750.0)
if fu < fu_min or fu > fu_max:
    st.sidebar.error(f"fu must be between {fu_min} and {fu_max} MPa.")
    error = True

n = st.sidebar.number_input("Strain hardening parameter (n)", min_value=n_min, max_value=n_max, step=0.1, value=5.5)
if n < n_min or n > n_max:
    st.sidebar.error(f"n must be between {n_min} and {n_max}.")
    error = True

# Scale features
def scale_features(feature_values):
    try:
        return feature_scaler.transform([feature_values])
    except Exception as e:
        st.error(f"Error scaling features: {e}")
        return None

# Prediction
if st.sidebar.button("Predict"):
    if error:
        st.error("Please ensure all inputs are within the valid range.")
    else:
        feature_values = [D, L, t, e, f0_2, fu, n]
        scaled_input = scale_features(feature_values)
        if scaled_input is not None:
            scaled_result = ann_model.predict(scaled_input)[0]
            result = target_scaler.inverse_transform([[scaled_result]])[0][0]
            st.subheader("Predicted Buckling Resistance")
            st.write(f"**Nu (kN):** {result:.2f}")

# Footer
st.markdown("---")
st.info("Developed by Sina Sarfarazi, University of Naples Federico II, Italy.\nContact: sina.srfz@gmail.com")
