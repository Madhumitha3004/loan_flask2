import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Load dataset
df = pd.read_csv("data/train.csv")

# Drop Loan_ID
df.drop(columns=["Loan_ID"], inplace=True)

# Add synthetic fields
np.random.seed(42)
df['Credit_Score'] = np.random.randint(300, 850, df.shape[0])
df['Obligations'] = np.random.randint(1000, 15000, df.shape[0])
df['Existing_Loans'] = np.random.randint(0, 5, df.shape[0])
df['House_Ownership'] = np.random.choice(['Owned', 'Rented'], size=df.shape[0])

# Rename for clarity
df.rename(columns={
    'ApplicantIncome': 'Annual_Income',
    'LoanAmount': 'Loan_Amount',
    'Loan_Amount_Term': 'Loan_Term'
}, inplace=True)

# Target column
df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

# Handle missing values
df.fillna({
    'Gender': df['Gender'].mode()[0],
    'Married': df['Married'].mode()[0],
    'Dependents': '0',
    'Self_Employed': 'No',
    'Credit_History': 1.0,
    'Loan_Term': df['Loan_Term'].mode()[0],
}, inplace=True)

df['Loan_Amount'].fillna(df['Loan_Amount'].median(), inplace=True)

# Encode categorical
categorical_cols = ['Gender', 'Married', 'Education', 'Self_Employed',
                    'Property_Area', 'Dependents', 'House_Ownership']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Split features/target
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# Scale numeric
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model, scaler, and feature names
os.makedirs("models", exist_ok=True)
model.feature_names = list(X.columns)  # Save feature names manually
joblib.dump(model, "models/loan_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("✅ Model, scaler, and feature names saved.")
