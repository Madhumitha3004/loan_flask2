import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import plotly.graph_objects as go

API_URL = "https://loan-flask2.onrender.com"  # Update this if needed
PREDICT_URL = f"{API_URL}/api/predict"

# --------------------------
# SESSION STATE INIT
# --------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None


# --------------------------
# SIGNUP PAGE
# --------------------------
def signup_page():
    st.title("📝 Sign Up")
    username = st.text_input("Username", key="signup_user")
    email = st.text_input("Email")
    phone = st.text_input("Phone Number")
    password = st.text_input("Password", type="password", key="signup_pass")

    if st.button("Sign Up"):
        payload = {
            "username": username,
            "email": email,
            "phone": phone,
            "password": password
        }
        try:
            res = requests.post(f"{API_URL}/signup", json=payload)
            if res.status_code == 200:
                st.success("✅ Registered successfully! Please log in.")
            else:
                st.error("❌ " + res.json().get("message", "Signup failed"))
        except Exception as e:
            st.error(f"🚨 Error connecting to server: {e}")


# --------------------------
# LOGIN PAGE
# --------------------------
def login_page():
    st.title("🔐 Login")
    username = st.text_input("Username", key="login_user")
    password = st.text_input("Password", type="password", key="login_pass")

    if st.button("Login"):
        try:
            res = requests.post(f"{API_URL}/login", json={
                "username": username,
                "password": password
            })
            if res.status_code == 200 and res.json().get("success"):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.success("✅ Login successful")
                st.rerun()
            else:
                st.error("❌ " + res.json().get("message", "Login failed"))
        except Exception as e:
            st.error(f"🚨 Error connecting to server: {e}")


# --------------------------
# MAIN APP (LOAN PREDICTOR)
# --------------------------
def loan_prediction_app():
    st.title("🏦 Loan Approval Predictor")
    st.write(f"Welcome, **{st.session_state.username}**! 👋")
    if st.button("Logout 🔓"):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.success("Logged out.")
        st.rerun()

    # --- Sidebar Inputs ---
    st.sidebar.header("Applicant Details")
    name = st.sidebar.text_input("Name *")
    age = st.sidebar.slider("Age *", 18, 70, 30)
    gender = st.sidebar.selectbox("Gender *", ["Male", "Female"])
    married = st.sidebar.selectbox("Marital Status *", ["Yes", "No"])
    dependents = st.sidebar.selectbox("Number of Dependents *", ["0", "1", "2", "3+"])
    education = st.sidebar.selectbox("Education *", ["Graduate", "Not Graduate"])
    self_employed = st.sidebar.selectbox("Self Employed *", ["Yes", "No"])
    employment_status = st.sidebar.selectbox("Employment Type *", ["Salaried", "Self-employed", "Business"])
    property_area = st.sidebar.selectbox("Property Area *", ["Urban", "Semiurban", "Rural"])
    house_ownership = st.sidebar.selectbox("House Ownership *", ["Owned", "Rented"])

    annual_income = st.sidebar.number_input("Applicant Annual Income (₹) *", min_value=0)
    coapplicant_income = st.sidebar.number_input("Co-applicant Income (₹)", min_value=0)
    loan_amount = st.sidebar.number_input("Requested Loan Amount (₹) *", min_value=0)
    loan_term = st.sidebar.slider("Loan Term (in months) *", 12, 360, 180)

    interest_rate = st.sidebar.slider("Preferred Interest Rate (%)", 6.0, 20.0, 9.0, step=0.1)

    credit_history = st.sidebar.selectbox("Credit History *", ["No Defaults", "Defaults"])
    credit_score = st.sidebar.slider("Credit Score *", 300, 850, 700)
    obligations = st.sidebar.number_input("Monthly Obligations (₹)", min_value=0)
    existing_loans = st.sidebar.slider("Existing Loan Accounts", 0, 10, 0)

    mandatory_fields = [name, annual_income, loan_amount, credit_score]
    if any(field == "" or field == 0 for field in mandatory_fields):
        st.error("❌ Please fill in all mandatory fields.")
        st.stop()

    # --- Prepare input dictionary for API ---
    input_data = {
        "Annual_Income": annual_income,
        "Loan_Amount": loan_amount,
        "Loan_Term": loan_term,
        "Credit_History": 1.0 if credit_history == "No Defaults" else 0.0,
        "Credit_Score": credit_score,
        "Obligations": obligations,
        "Existing_Loans": existing_loans,
        "Gender_Male": 1 if gender == "Male" else 0,
        "Married_Yes": 1 if married == "Yes" else 0,
        "Education_Graduate": 1 if education == "Graduate" else 0,
        "Self_Employed_Yes": 1 if self_employed == "Yes" else 0,
        "Property_Area_Urban": 1 if property_area == "Urban" else 0,
        "Property_Area_Semiurban": 1 if property_area == "Semiurban" else 0,
        "Dependents_1": 1 if dependents == "1" else 0,
        "Dependents_2": 1 if dependents == "2" else 0,
        "Dependents_3+": 1 if dependents == "3+" else 0,
        "House_Ownership_Rented": 1 if house_ownership == "Rented" else 0,
    }

    expected_features = [
        'Annual_Income', 'Loan_Amount', 'Loan_Term', 'Credit_History', 'Credit_Score',
        'Obligations', 'Existing_Loans', 'Gender_Male', 'Married_Yes', 'Education_Graduate',
        'Self_Employed_Yes', 'Property_Area_Urban', 'Property_Area_Semiurban',
        'Dependents_1', 'Dependents_2', 'Dependents_3+', 'House_Ownership_Rented'
    ]
    for col in expected_features:
        if col not in input_data:
            input_data[col] = 0

    if st.button("Predict Loan Status"):
        try:
            response = requests.post(PREDICT_URL, json=input_data)
            result = response.json()
            prediction = result["prediction"]
            probability = result["probability"]

            if prediction == 1:
                st.success(f"✅ {name}, your loan is **Approved**!")
                st.write(f"**Approval Probability:** {probability * 100:.2f}%")
                monthly_income = (annual_income + coapplicant_income) / 12
                disposable_income = max(monthly_income - obligations, 0)
                affordable_emi = disposable_income * 0.5
                R = interest_rate / 100 / 12
                N = loan_term

                if R > 0:
                    eligible_amount = (affordable_emi * ((1 + R) ** N - 1)) / (R * (1 + R) ** N)
                else:
                    eligible_amount = affordable_emi * N

                eligible_emi = affordable_emi
                eligible_total = eligible_emi * N
                eligible_interest = eligible_total - eligible_amount

                if R > 0:
                    requested_emi = (loan_amount * R * (1 + R) ** N) / ((1 + R) ** N - 1)
                else:
                    requested_emi = loan_amount / N

                requested_total = requested_emi * N
                requested_interest = requested_total - loan_amount

                st.markdown("---")
                st.subheader("📉 Requested Loan Breakdown")
                st.write(f"**Requested EMI:** ₹{requested_emi:,.0f}")
                st.write(f"**Total Payable:** ₹{requested_total:,.0f}")
                st.write(f"**Total Interest:** ₹{requested_interest:,.0f}")

                fig1, ax1 = plt.subplots(figsize=(3.5, 3.5))
                ax1.pie([loan_amount, requested_interest], labels=['Principal', 'Interest'],
                        autopct='%1.1f%%', startangle=90, colors=['#4CAF50', '#FF7043'])
                ax1.axis('equal')
                st.pyplot(fig1)

                st.markdown("---")
                st.subheader("💰 Estimated Eligible Loan Breakdown")
                st.write(f"**Eligible Loan Amount:** ₹{eligible_amount:,.0f}")
                st.write(f"**Estimated EMI:** ₹{eligible_emi:,.0f}")
                st.write(f"**Total Payable:** ₹{eligible_total:,.0f}")
                st.write(f"**Total Interest:** ₹{eligible_interest:,.0f}")

                fig2, ax2 = plt.subplots(figsize=(3.5, 3.5))
                ax2.pie([eligible_amount, eligible_interest], labels=['Principal', 'Interest'],
                        autopct='%1.1f%%', startangle=90, colors=['#4CAF50', '#FF7043'])
                ax2.axis('equal')
                st.pyplot(fig2)

                st.markdown("---")
                st.subheader("📊 EMI Payment Schedule (Yearly Breakdown)")
                balance = eligible_amount
                schedule = []

                for month in range(1, N + 1):
                    interest_payment = balance * R
                    principal_payment = eligible_emi - interest_payment
                    balance -= principal_payment
                    schedule.append({
                        "Month": month,
                        "Year": (month - 1) // 12 + 1,
                        "EMI": eligible_emi,
                        "Principal": max(principal_payment, 0),
                        "Interest": max(interest_payment, 0),
                        "Balance": max(balance, 0)
                    })

                df_schedule = pd.DataFrame(schedule)
                df_yearly = df_schedule.groupby("Year").agg({
                    "EMI": "sum", "Principal": "sum", "Interest": "sum", "Balance": "last"
                }).reset_index()

                fig = go.Figure()
                fig.add_trace(go.Bar(x=df_yearly["Year"], y=df_yearly["Principal"], name="Principal", marker_color="#4CAF50"))
                fig.add_trace(go.Bar(x=df_yearly["Year"], y=df_yearly["Interest"], name="Interest", marker_color="#FF7043"))
                fig.update_layout(
                    barmode="stack", xaxis_title="Year", yaxis_title="Amount (₹)",
                    title="Yearly Principal & Interest Breakdown (Eligible Loan)",
                    legend_title="Component", height=500
                )
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("📅 Yearly Payment Table"):
                    st.dataframe(df_yearly.style.format({
                        "EMI": "₹{:,.0f}", "Principal": "₹{:,.0f}",
                        "Interest": "₹{:,.0f}", "Balance": "₹{:,.0f}"
                    }), use_container_width=True)

            else:
                st.error(f"❌ Sorry {name}, your loan is **Rejected**.")
                st.write("### Rejection Reasons:")
                reasons = []
                if annual_income < 500000:
                    reasons.append("Insufficient annual income.")
                if credit_score < 600:
                    reasons.append("Low credit score.")
                if loan_amount > (annual_income * 10):
                    reasons.append("Requested loan amount exceeds eligible limit.")
                if credit_history == "Defaults":
                    reasons.append("Credit history shows defaults.")
                for r in reasons:
                    st.write(f"- {r}")

        except Exception as e:
            st.error(f"🚨 Failed to connect to prediction API: {e}")

# --------------------------
# MAIN APP ROUTER
# --------------------------
if not st.session_state.authenticated:
    page = st.sidebar.selectbox("Choose", ["Login", "Sign Up"])
    if page == "Login":
        login_page()
    else:
        signup_page()
else:
    loan_prediction_app()
