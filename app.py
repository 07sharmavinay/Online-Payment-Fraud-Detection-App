import streamlit as st
import numpy as np
import joblib

# Page configuration
st.set_page_config(page_title="Fraud Detection", page_icon="💳", layout="centered")

# Load model
try:
    model = joblib.load("Random_forest_model.pkl")
except FileNotFoundError:
    st.error("❌ Model file not found. Make sure 'decision_tree_model.pkl' is in the app directory.")
    model = None

# Mapping for transaction types
type_mapping = {"CASH_OUT": 1, "PAYMENT": 2, "CASH_IN": 3, "TRANSFER": 4, "DEBIT": 5}

# Title section
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>💳 Online Fraud Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Detect potentially fraudulent transactions in real time.</p>", unsafe_allow_html=True)
st.markdown("---")

if model is not None:
    # Input section
    with st.form("fraud_form"):
        st.subheader("🔍 Transaction Details")

        col1, col2 = st.columns(2)
        with col1:
            transaction_type = st.selectbox("📂 Transaction Type", list(type_mapping.keys()))
            amount = st.number_input("💰 Amount", min_value=0.0, format="%.2f")
        with col2:
            old_balance_orig = st.number_input("🏦 Old Balance (Originator)", min_value=0.0, format="%.2f")
            new_balance_orig = st.number_input("🏦 New Balance (Originator)", min_value=0.0, format="%.2f")

        submitted = st.form_submit_button("🔎 Predict")

    # Prediction
    if submitted:
        type_numeric = type_mapping[transaction_type]
        features = np.array([[type_numeric, amount, old_balance_orig, new_balance_orig]])

        try:
            prediction = model.predict(features)

            st.markdown("---")
            if prediction[0] == 1:
                st.error("⚠️ **Alert: This transaction is likely FRAUDULENT.**", icon="🚨")
                st.markdown("Please investigate further before processing.")
            else:
                st.success("✅ **This transaction appears to be legitimate.**", icon="🔒")
        except Exception as e:
            st.error(f"Prediction error: {e}")
else:
    st.warning("⚠️ Model could not be loaded. Predictions are disabled.")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; font-size: 12px;'>Built with ❤️ using Streamlit</p>",
    unsafe_allow_html=True
)
