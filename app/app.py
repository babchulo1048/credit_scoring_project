import streamlit as st
import pandas as pd
import joblib
import shap
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load model and explainer
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
explainer = joblib.load(os.path.join(BASE_DIR, "shap_explainer.pkl"))

# Initialize LabelEncoder (must match training)
label_encoders = {}
label_cols = ['Sex', 'Job', 'Housing', 'Saving_accounts', 'Checking_account', 'Purpose']
for col in label_cols:
    label_encoders[col] = LabelEncoder()
    # Fit with possible values (adjust based on your dataset)
    if col == 'Sex':
        label_encoders[col].fit(['male', 'female'])
    elif col == 'Job':
        label_encoders[col].fit(['unskilled', 'skilled', 'highly_skilled'])  # Adjust as needed
    elif col == 'Housing':
        label_encoders[col].fit(['own', 'rent', 'free'])
    elif col == 'Saving_accounts':
        label_encoders[col].fit(['little', 'moderate', 'quite rich', 'rich', 'unknown'])
    elif col == 'Checking_account':
        label_encoders[col].fit(['little', 'moderate', 'rich', 'unknown'])
    elif col == 'Purpose':
        label_encoders[col].fit(['car', 'furniture', 'radio/TV', 'education', 'business', 'repairs', 'vacation/others'])

# Streamlit app
st.title("Credit Scoring Demo")

st.header("Enter Applicant Details")
age = st.number_input("Age", min_value=18, max_value=100, value=32)
sex = st.selectbox("Sex", ['male', 'female'], index=0)
job = st.selectbox("Job", ['unskilled', 'skilled', 'highly_skilled'], index=1)
housing = st.selectbox("Housing", ['own', 'rent', 'free'], index=1)
saving_accounts = st.selectbox("Saving Accounts", ['little', 'moderate', 'quite rich', 'rich', 'unknown'], index=0)
checking_account = st.selectbox("Checking Account", ['little', 'moderate', 'rich', 'unknown'], index=1)
credit_amount = st.number_input("Credit Amount (Birr)", min_value=100, value=3000)
duration = st.number_input("Duration (Months)", min_value=1, value=24)
purpose = st.selectbox("Purpose", ['car', 'furniture', 'radio/TV', 'education', 'business', 'repairs', 'vacation/others'], index=0)

if st.button("Predict"):
    # Prepare input
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'Job': [job],
        'Housing': [housing],
        'Saving_accounts': [saving_accounts],
        'Checking_account': [checking_account],
        'Credit_amount': [credit_amount],
        'Duration': [duration],
        'Purpose': [purpose],
        'Monthly_Payment': [credit_amount / duration]
    })

    # Encode categorical features
    for col in label_cols:
        input_data[col] = label_encoders[col].transform(input_data[col])

    # Predict
    risk_prob = model.predict_proba(input_data)[:, 1][0]
    # Convert to credit score (300–900 scale)
    credit_score = 900 - (risk_prob * 600)  # Linear mapping: 0 risk = 900, 1 risk = 300
    decision = "Approve" if credit_score >= 700 else "Maybe" if credit_score >= 600 else "Caution" if credit_score >= 500 else "Reject"

    # SHAP explanation
    shap_values = explainer(input_data)
    shap_values_class = shap_values[:, :, 1]
    top_factors = pd.Series(shap_values_class.values[0], index=input_data.columns).abs().sort_values(ascending=False).index[:3].tolist()

    # Display results
    st.header("Prediction Results")
    st.write(f"**Risk Probability**: {risk_prob:.2%}")
    st.write(f"**Credit Score**: {int(credit_score)}")
    st.write(f"**Decision**: {decision}")
    st.write(f"**Top Factors**: {top_factors}")

    # SHAP plot
    st.subheader("SHAP Explanation")
    shap.plots.bar(shap_values_class[0], show=False)
    st.pyplot(plt.gcf())
