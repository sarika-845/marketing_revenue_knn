import streamlit as st
import joblib

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Page setup
st.set_page_config(page_title="Marketing Revenue Predictor", page_icon="💹", layout="centered")

# Title & description
st.title("💼 Marketing Revenue Prediction")
st.markdown("""
Welcome to the **Marketing Revenue Predictor** dashboard.  
Enter your campaign details below and get instant revenue predictions.
""")

# Inputs
ad_spend = st.number_input("💰 Ad Spend", value=100.0)
price = st.number_input("🏷️ Price", value=10.0)
discount_rate = st.number_input("🔻 Discount Rate", value=0.01)
market_reach = st.number_input("🌍 Market Reach", value=1000.0)
impressions = st.number_input("👀 Impressions", value=5000.0)
click_through_rate = st.number_input("🖱️ Click Through Rate", value=0.05)
competition_index = st.number_input("⚔️ Competition Index", value=1.0)
seasonality_index = st.number_input("📅 Seasonality Index", value=1.0)
campaign_duration_days = st.number_input("⏳ Campaign Duration (days)", value=30)
customer_lifetime_value = st.number_input("👤 Customer Lifetime Value", value=100.0)

# Features in correct order
features = [[ad_spend, price, discount_rate, market_reach,
             impressions, click_through_rate, competition_index,
             seasonality_index, campaign_duration_days,
             customer_lifetime_value]]

# Scale & predict
features_scaled = scaler.transform(features)
prediction = model.predict(features_scaled)

# Display result
st.success(f"🧾 Predicted Sales Revenue: **{prediction[0]:.2f}**")

# KPI card
st.metric(label="Predicted Revenue", value=f"{prediction[0]:.2f}", delta="vs last campaign")

# Optional chart (dummy comparison)
import pandas as pd
comparison = pd.DataFrame({
    "Actual": [100, 120, 140, 160],
    "Predicted": [110, 125, 135, 155]
})
st.line_chart(comparison)

