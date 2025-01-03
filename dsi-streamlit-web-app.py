
# Import libraries
import streamlit as st
import pandas as pd
import joblib

# Load model pipeline object
model = joblib.load("model.joblib")

# add and instructions
st.title("Purchase Prediction Model")
st.subheader("Enter customer information and submit likelihood to purchase")

# age input form
age = st.number_input(
    label= "01. Enter Customer's Age",
    min_value= 18,
    max_value= 120,
    value = 35
    )# predetermined age

# gender input form
gender = st.radio(
    label = "02. Enter Customer's Gender",
    options = ['M','F']
    )

# credit score input form
credit_score = st.number_input(
    label= "03. Enter customer's Credit Score",
    min_value= 0,
    max_value= 1000,
    value = 500
    )

# submit inputs to model
if st.button("Submit For Prediction"):
    
    # store data in dataframe for prediction
    new_data = pd.DataFrame({"age": [age], "gender" : [gender], "credit_score":[credit_score]})
    
    # apply model pipeline to input data and extract probability prediction
    pred_proba = model.predict_proba(new_data)[0][1]
    
    # output prediction
    st.subheader(f"Based on these customer attributes, the model predicts a purchase probability of {pred_proba:.0%}")













