import streamlit as st
import numpy as np
import joblib
model=joblib.load("churn_prediction.pkl")
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
st.title("CUSTOMER CHURN PREDICTION")
st.markdown("Enter the details of customer below")
credit_score=st.number_input("credit score", min_value=300, max_value=900, value=650)
gender=st.selectbox("gender", ["Male", "Female"])
gender_binary=1 if gender=="Male" else 0
age=st.number_input("age", min_value=18, max_value=100, value=24)
tenure = st.slider("Tenure (Years with Bank)", 0, 10, 3)
balance=st.number_input("Account_balance", min_value=0.0, max_value=2500000.0, value=18000.0)
products=st.slider("number of products", min_value=1, max_value=4, value=2)
card = 1 if st.selectbox("Has Credit Card?", ["Yes", "No"]) == "Yes" else 0
activie=st.selectbox("Is active member?",["yes","no"])
active_binary=1 if activie=="yes" else 0

salary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)
geography=st.selectbox("Geographyh", ["France", "Spain", "Germany"])
geography_germany=1 if geography=="Germany" else 0
geography_spain=1 if geography=="Spain" else 0
model_input = np.array([[credit_score, gender_binary, age, tenure, balance, products, card, active_binary, salary, geography_germany, geography_spain]])
if st.button("Predict Churn"):
    prediction=model.predict(model_input)[0]
    if prediction==1:
        st.error("This Customer is likely to Churn")
    else:
        st.success("This Custoer is likely to Stay")