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

# Set page configuration
st.set_page_config(
    page_title="Buckling Resistance Prediction",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown(
    """
    <style>
    body {
        background-color: #f5f5f5;
        color: #333;
        font-family: 'Arial', sans-serif;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stSidebar .css-1d391kg {
        background-color: #ececec;
    }
    .main-title {
        text-align: center;
        color: #003366;
        font-size: 36px;
        margin-bottom: 20px;
    }
    .description {
        font-size: 18px;
        color: #555;
        margin-bottom: 30px;
    }
    .prediction {
        font-size: 24px;
        color: #4CAF50;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and Description
st.markdown("<h1 class='main-title'>Buckling Resistance Prediction Using ANN</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='description'>This app predicts the buckling resistance (Nu) of CHS SS members based on user input parameters using an Artificial Neural Network (ANN) model.</p>",
    unsafe_allow_html=True
)

# Sidebar for Input Features
st.sidebar.header("Input Features")
inputs = {}
with st.sidebar:
    for feature in [
        "Diameter (D) (mm):",
        "Length (L) (mm):",
        "Thickness (t) (mm):",
        "Edge distance (e) (mm):",
        "Proof strength (f0.2) (MPa):",
        "Ultimate strength (fu) (MPa):",
        "Strain hardening parameter (n):"
    ]:
        label = feature.split(" (")[0]
        inputs[label] = st.number_input(f"{feature}", value=1.0, step=0.1)

# Function to Scale Features
def scale_features(feature_values):
    try:
        scaled_values = feature_scaler.transform([feature_values])
        return scaled_values
    except Exception as e:
        st.error(f"Error scaling features: {e}")
        return None

# Main Content Area
with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Predict"):
            if ann_model is None:
                st.error("Model not loaded. Please check your setup.")
            else:
                feature_values = [inputs[name.split(" (")[0]] for name in feature_names]
                scaled_input = scale_features(feature_values)
                if scaled_input is not None:
                    scaled_result = ann_model.predict(scaled_input)[0]
                    result = target_scaler.inverse_transform([[scaled_result]])[0][0]
                    st.markdown(f"<p class='prediction'>Predicted Buckling Resistance (Nu): {result:.2f} kN</p>", unsafe_allow_html=True)

    with col2:
        if st.button("Save to History"):
            if ann_model:
                feature_values = [inputs[name.split(" (")[0]] for name in feature_names]
                scaled_input = scale_features(feature_values)
                if scaled_input is not None:
                    scaled_result = ann_model.predict(scaled_input)[0]
                    result = target_scaler.inverse_transform([[scaled_result]])[0][0]
                    if "history" not in st.session_state:
                        st.session_state.history = []
                    st.session_state.history.append({
                        "features": inputs,
                        "Load (kN)": result
                    })
                st.success("Prediction saved to history.")

# Display Prediction History
if "history" in st.session_state and st.session_state.history:
    st.subheader("Prediction History")
    history_df = pd.DataFrame(st.session_state.history)
    st.write(history_df)

    # Plot History
    if st.button("Plot History"):
        loads = [record['Load (kN)'] for record in st.session_state.history]
        plt.figure(figsize=(8, 5))
        plt.plot(loads, marker='o', linestyle='-', label='Load (kN)')
        plt.title("Prediction History")
        plt.xlabel("Prediction Count")
        plt.ylabel("Load (kN)")
        plt.grid(True)
        plt.legend()
        st.pyplot(plt)

# Export to CSV
if st.button("Save Results to CSV"):
    try:
        csv_path = 'prediction_results.csv'
        pd.DataFrame(st.session_state.history).to_csv(csv_path, index=False)
        st.success(f"Results saved to {csv_path}")
    except Exception as e:
        st.error(f"Error saving results: {e}")

# Footer
st.markdown("---")
st.info("Developed by Sina Sarfarazi, University of Naples Federico II, Italy.\nContact: sina.srfz@gmail.com")
