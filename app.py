import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Telecom Churn Predictor", page_icon="📉")


@st.cache_resource
def load_bundle():
    return joblib.load("models/churn_model.pkl")


bundle = load_bundle()
pipeline = bundle["pipeline"]
threshold = bundle["threshold"]
columns = bundle["columns"]

st.title("📉 Telecom Customer Churn Predictor")
st.write("Enter the customer's details to estimate their churn risk.")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly = st.number_input("Monthly Charges", 18.0, 120.0, 70.0)
    total = st.number_input("Total Charges", 0.0, 9000.0, 800.0)

with col2:
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multiple = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

if st.button("Predict Churn"):
    row = {
        "gender": gender, "SeniorCitizen": senior, "Partner": partner,
        "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
        "MultipleLines": multiple, "InternetService": internet,
        "OnlineSecurity": security, "OnlineBackup": backup,
        "DeviceProtection": protection, "TechSupport": support,
        "StreamingTV": tv, "StreamingMovies": movies, "Contract": contract,
        "PaperlessBilling": paperless, "PaymentMethod": payment,
        "MonthlyCharges": monthly, "TotalCharges": total,
    }
    input_df = pd.DataFrame([row])

    # Recreate the same engineered features used during training
    input_df["TenureGroup"] = pd.cut(
        input_df["tenure"], bins=[-1, 12, 24, 36, 48, 60, 72],
        labels=["0-12", "13-24", "25-36", "37-48", "49-60", "61-72"])
    service_cols = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    input_df["TotalServices"] = (input_df[service_cols] == "Yes").sum(axis=1)

    input_df = input_df[columns]   # Reorder columns to match the training data
    proba = pipeline.predict_proba(input_df)[:, 1][0]

    st.metric("Churn Probability", f"{proba:.1%}")
    st.progress(float(proba))
    if proba >= threshold:
        st.error("⚠️ High churn risk: consider offering this customer a retention deal.")
    else:
        st.success("✅ Low churn risk: this customer is likely to stay.")
    st.caption(f"Decision threshold: {threshold}")