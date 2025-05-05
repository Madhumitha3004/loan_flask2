import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and scaler
model = joblib.load("models/loan_model.pkl")
scaler = joblib.load("models/scaler.pkl")
model_features = model.feature_names

st.set_page_config(page_title="Loan Approval Prediction App", layout="wide")
st.title("🏦 Loan Approval Prediction App")

st.markdown("Enter your personal and financial details to check your loan approval status.")

# Sidebar inputs
with st.sidebar:
    st.header("📋 Applicant Information")
    name = st.text_input("Name")
    age = st.slider("Age", 18, 70, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Marital Status", ["Married", "Single"])
    dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
    employment = st.selectbox("Employment Status", ["Employed", "Self_Employed"])
    income = st.number_input("Applicant Annual Income (INR)", min_value=10000, step=1000)
    credit_score = st.slider("Applicant Credit Score", 300, 850, 650)
    obligations = st.number_input("Monthly Obligations (EMI + Debt)", min_value=0)
    existing_loans = st.number_input("Number of Existing Loan Accounts", min_value=0, step=1)
    loan_amount = st.number_input("Desired Loan Amount (INR)", min_value=10000)
    loan_term = st.selectbox("Loan Term (in months)", [12, 36, 60, 120, 180, 240, 300, 360])
    credit_history = st.selectbox("Credit History", [1.0, 0.0])
    house = st.selectbox("House Ownership", ["Owned", "Rented"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    has_coapplicant = st.checkbox("Add Coapplicant?")
    co_income = 0
    co_credit = 0
    if has_coapplicant:
        co_income = st.number_input("Coapplicant Annual Income (INR)", min_value=0)
        co_credit = st.slider("Coapplicant Credit Score", 300, 850, 700)

# Combine incomes
total_income = income + co_income
avg_credit_score = (credit_score + co_credit) / (2 if has_coapplicant else 1)

# Input dictionary
input_dict = {
    'Age': age,
    'Annual_Income': total_income,
    'Credit_Score': avg_credit_score,
    'Obligations': obligations,
    'Existing_Loans': existing_loans,
    'Loan_Amount': loan_amount,
    'Loan_Term': loan_term,
    'Credit_History': credit_history,
    'Gender_Male': 1 if gender == "Male" else 0,
    'Married_Yes': 1 if married == "Married" else 0,
    'Education_Not Graduate': 1 if education == "Not Graduate" else 0,
    'Self_Employed_Yes': 1 if employment == "Self_Employed" else 0,
    'Property_Area_Semiurban': 1 if property_area == "Semiurban" else 0,
    'Property_Area_Urban': 1 if property_area == "Urban" else 0,
    'Dependents_1': 1 if dependents == "1" else 0,
    'Dependents_2': 1 if dependents == "2" else 0,
    'Dependents_3+': 1 if dependents == "3+" else 0,
    'House_Ownership_Owned': 1 if house == "Owned" else 0
}

# Fill missing features
for col in model_features:
    if col not in input_dict:
        input_dict[col] = 0

input_df = pd.DataFrame([input_dict])[model_features]
input_scaled = scaler.transform(input_df)

# Prediction
if st.button("🔍 Predict Loan Status"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("📊 Prediction Result")
    if prediction == 1:
        st.success("✅ Your loan is likely to be **Approved**.")
    else:
        st.error("❌ Your loan is likely to be **Rejected**.")

    st.markdown(f"**Approval Probability:** `{probability:.2%}`")

    # Calculate eligible amount
    monthly_income = total_income / 12
    max_emi = max((monthly_income - obligations) * 0.5, 0)

    annual_interest_rate = 0.10
    monthly_interest_rate = annual_interest_rate / 12
    n = loan_term

    if monthly_interest_rate > 0:
        emi_factor = (monthly_interest_rate * (1 + monthly_interest_rate)**n) / ((1 + monthly_interest_rate)**n - 1)
        eligible_amount = max_emi / emi_factor
    else:
        eligible_amount = max_emi * n

    max_cap = total_income * 5
    eligible_amount = min(eligible_amount, max_cap)

    if prediction == 1:
        st.markdown(f"**Estimated Eligible Loan Amount:** ₹`{eligible_amount:,.0f}`")

    # Feature importance
    st.subheader("📈 Feature Importance")
    importance = model.feature_importances_
    fig, ax = plt.subplots(figsize=(6, 4))
    sorted_idx = np.argsort(importance)[::-1]
    sorted_features = [model_features[i] for i in sorted_idx]
    sorted_importance = importance[sorted_idx]

    sns.barplot(x=sorted_importance[:10], y=sorted_features[:10], ax=ax, palette="viridis")
    ax.set_title("Top 10 Important Features")
    st.pyplot(fig)

    # Visualization 1: Gender Distribution Comparison (Pie chart)
    gender_data = ["Male", "Female"]  # Sample gender data for comparison (could be your dataset distribution)
    gender_counts = [0.55, 0.45]  # Example percentage of males/females
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(gender_counts, labels=gender_data, autopct='%1.1f%%', colors=['#66b3ff', '#ff9999'])
    ax.set_title(f"Gender Distribution (You are {gender})")
    st.pyplot(fig)

    # Visualization 2: Employment Status Distribution (Pie chart)
    employment_data = ["Employed", "Self_Employed"]  # Sample employment data
    employment_counts = [0.75, 0.25]  # Example distribution
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(employment_counts, labels=employment_data, autopct='%1.1f%%', colors=['#99ff99', '#ffcc99'])
    ax.set_title(f"Employment Status Distribution (You are {employment})")
    st.pyplot(fig)

    # Visualization 3: Loan Term Distribution (Bar chart)
    loan_terms = [12, 36, 60, 120, 180, 240, 300, 360]
    loan_term_distribution = [0.1, 0.2, 0.3, 0.05, 0.1, 0.1, 0.1, 0.15]  # Example distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(loan_terms, loan_term_distribution, color='skyblue')
    ax.set_xlabel("Loan Term (Months)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Loan Term Distribution (You requested {loan_term} months)")
    st.pyplot(fig)

    # Visualization 4: Credit History Comparison (Bar chart)
    credit_history_data = [1.0, 0.0]  # Example of credit history distributions
    credit_history_counts = [0.8, 0.2]  # Example distribution for approvals and rejections
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar(credit_history_data, credit_history_counts, color=['#4CAF50', '#f44336'])
    ax.set_xlabel("Credit History")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Credit History Distribution (Your credit history: {credit_history})")
    st.pyplot(fig)
