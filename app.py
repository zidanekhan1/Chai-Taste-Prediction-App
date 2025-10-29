import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Chai Taste Predictor", layout="centered")
st.title("☕ Chai Taste Predictor using ANN")
st.write("Predict whether a cup of chai will taste **Good** or **Not Good** based on its ingredients.")

@st.cache_resource
def load_all():
    model = load_model("chai_model.h5")
    scaler = joblib.load("scaler.pkl")
    encoders = joblib.load("encoders.pkl")
    return model, scaler, encoders

try:
    model, scaler, encoders = load_all()
    st.success("✅ Model and preprocessing tools loaded successfully!")
except Exception as e:
    st.error(f"❌ Could not load model/scaler/encoders. Error: {e}")
    st.stop()

st.header("Enter Your Chai Details 🍵")

col1, col2 = st.columns(2)

with col1:
    sugar_level = st.slider("Sugar Level (0–5 spoons)", 0, 5, 3)
    masala_level = st.slider("Masala Level (0–5 spoons)", 0, 5, 2)
    masala_type = st.selectbox("Masala Type", options=["Cardamom", "Ginger", "Cinnamon", "Mixed"])

with col2:
    base_type = st.selectbox("Base Type", options=["Water", "Milk", "Both"])
    base_quantity_ml = st.number_input("Base Quantity (ml)", min_value=100, max_value=300, value=200, step=10)
    brew_time_min = st.slider("Brew Time (minutes)", 1, 10, 5)

st.markdown("---")

if st.button("Predict Taste"):
    
    input_df = pd.DataFrame([[
        sugar_level, masala_level, masala_type, base_type, base_quantity_ml, brew_time_min
    ]], columns=["sugar_level", "masala_level", "masala_type", "base_type", "base_quantity_ml", "brew_time_min"])

    
    for col in ["masala_type", "base_type"]:
        le = encoders[col]
        try:
            input_df[col] = le.transform(input_df[col])
        except ValueError:
            st.error(f"The selected value '{input_df[col].iloc[0]}' for '{col}' was not seen during training.")
            st.stop()

    
    scaled_sample = scaler.transform(input_df)

    
    prediction = model.predict(scaled_sample)[0][0]
    label = "Good ☕" if prediction >= 0.5 else "Not Good 😕"

    st.subheader("Prediction Result")
    st.metric("Predicted Taste", label)
    st.write(f"Confidence: {prediction*100:.2f}%")

    if prediction >= 0.8:
        st.success("This chai looks delicious — high confidence it's GOOD! 🍵")
    elif prediction >= 0.5:
        st.info("This chai might taste okay, but could go either way 😌")
    else:
        st.warning("Hmm... might need better balance in ingredients 😕")

st.markdown("---")

st.caption("Trained ANN Model • Files: chai_model.h5 | scaler.pkl | encoders.pkl")
