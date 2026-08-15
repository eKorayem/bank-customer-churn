import streamlit as st
import sys
import os
import pandas as pd
import joblib

# Import your custom class so joblib can rebuild the pipeline
# from scripts.preprocessing import ChurnFeatureEngineer
import __main__

# Add the 'scripts' folder to Python's internal path
sys.path.append(os.path.abspath("scripts"))
from preprocessing import ChurnFeatureEngineer

# The Fix: Manually inject the class into the "main" environment
__main__.ChurnFeatureEngineer = ChurnFeatureEngineer

from db_utils import get_db_connection

# 1. Page Configuration
st.set_page_config(page_title="Bank Churn Predictor", page_icon="🏦", layout="centered")
st.title("Customer Churn Prediction App")
st.write("Enter the customer's details below to predict their likelihood of leaving the bank.")

# 2. Load the Pipeline and Model
@st.cache_resource
def load_assets():
    # Load the preprocessing pipeline (which contains the robust scaler and one-hot encoder)
    data = joblib.load('data/processed/dataset_bundle.pkl')
    pipeline = data['pipeline']
    # Load the trained Random Forest model
    model = joblib.load('data/processed/champion_model.pkl')
    return pipeline, model

pipeline, model = load_assets()

# 3. Sidebar for User Input
st.sidebar.header("Customer Details")

def user_input_features():
    credit_score = st.sidebar.slider("Credit Score", 300, 850, 600)
    geography = st.sidebar.selectbox("Country", ["France", "Germany", "Spain"])
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    age = st.sidebar.slider("Age", 18, 100, 40)
    tenure = st.sidebar.slider("Tenure (Years)", 0, 10, 5)
    balance = st.sidebar.number_input("Account Balance ($)", 0.0, 300000.0, 50000.0)
    num_of_products = st.sidebar.slider("Number of Products", 1, 4, 1)
    has_cr_card = st.sidebar.selectbox("Has Credit Card?", [1, 0])
    is_active_member = st.sidebar.selectbox("Is Active Member?", [1, 0])
    estimated_salary = st.sidebar.number_input("Estimated Salary ($)", 0.0, 200000.0, 60000.0)

    # Format into a Pandas DataFrame exactly like your original raw data
    data = {
        'CreditScore': credit_score,
        'Geography': geography,
        'Gender': gender,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumProducts': num_of_products,
        'HasCreditCard': has_cr_card,
        'IsActive': is_active_member,
        'Salary': estimated_salary
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Display the user's input on the main page
st.subheader("Customer Profile")
st.dataframe(input_df)

# 4. Prediction Logic
if st.button("Predict Churn"):
    # Pass the raw data through the pipeline (adds BalanceSalaryRatio, scales, encodes)
    processed_data = pipeline.transform(input_df)
    
    # Get the prediction and the probability percentage
    prediction = model.predict(processed_data)
    probability = model.predict_proba(processed_data)[0][1] * 100

    st.subheader("Prediction Results")
    if prediction[0] == 1:
        st.error(f"**High Risk of Churn!** (Probability: {probability:.1f}%)")
        st.write("Recommendation: Route this customer to the retention team immediately.")
    else:
        st.success(f"**Customer is Likely to Stay** (Churn Probability: {probability:.1f}%)")