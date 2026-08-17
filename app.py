import streamlit as st
import sys
import os
import pandas as pd
import joblib
import plotly.graph_objects as go

# Import your custom class so joblib can rebuild the pipeline
import __main__

# Add the 'scripts' folder to Python's internal path
sys.path.append(os.path.abspath("scripts"))
from preprocessing import ChurnFeatureEngineer

# Manually inject the class into the "main" environment
__main__.ChurnFeatureEngineer = ChurnFeatureEngineer

# 1. Page Configuration (Set to wide layout for dashboards)
st.set_page_config(page_title="Bank Churn Predictor", page_icon="🏦", layout="wide")
st.title("Customer Churn Prediction Dashboard")
st.markdown("Enter the customer's details in the sidebar to dynamically calculate their churn risk.")

# 2. Load the Pipeline and Model
@st.cache_resource
def load_assets():
    data = joblib.load('data/processed/dataset_bundle.pkl')
    pipeline = data['pipeline']
    model = joblib.load('data/processed/champion_model.pkl')
    return pipeline, model

pipeline, model = load_assets()

# 3. Sidebar for User Input
st.sidebar.header("Customer Details")

def user_input_features():
    geography = st.sidebar.selectbox("Country", ["France", "Germany", "Spain", "UK", "Canada", "USA"])
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    age = st.sidebar.slider("Age", 18, 100, 40)
    tenure = st.sidebar.slider("Tenure (Years)", 0, 10, 5)
    balance = st.sidebar.number_input("Account Balance ($)", 0.0, 300000.0, 50000.0)
    num_of_products = st.sidebar.slider("Number of Products", 1, 4, 1)
    has_cr_card = st.sidebar.selectbox("Has Credit Card?", [1, 0])
    is_active_member = st.sidebar.selectbox("Is Active Member?", [1, 0])
    estimated_salary = st.sidebar.number_input("Estimated Salary ($)", 0.0, 200000.0, 60000.0)

    data = {
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

# 4. Interactive Gauge Chart Function
def create_gauge_chart(probability):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Churn Probability Risk", 'font': {'size': 20}},
        number={'suffix': "%", 'font': {'size': 40}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "black"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': "#00cc96"},    # Safe / Green
                {'range': [30, 70], 'color': "#ffa15a"},   # Warning / Orange
                {'range': [70, 100], 'color': "#ef553b"}   # Danger / Red
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': probability
            }
        }
    ))
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# 5. UI Layout (Two Columns)
col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("Current Customer Profile")
    st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)

with col2:
    st.subheader("Real-Time Prediction")
    
    # Process data and predict
    processed_data = pipeline.transform(input_df)
    prediction = model.predict(processed_data)
    probability = model.predict_proba(processed_data)[0][1] * 100

    # Render Plotly Gauge
    st.plotly_chart(create_gauge_chart(probability), use_container_width=True)
    
    # Render Text Alerts
    if prediction[0] == 1:
        st.error(f"**Action Required:** Customer has a high risk of churning.")
        st.info("Recommendation: Route this customer to the retention team to discuss loyalty incentives.")
    else:
        st.success(f"**Stable:** Customer is likely to stay.")
        st.info("Recommendation: Maintain standard communication. No immediate retention action required.")