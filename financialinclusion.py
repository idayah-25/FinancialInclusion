import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

model = joblib.load('financial_model.pkl')

df = pd.read_csv('Financial_inclusion_dataset.csv')

st.title('Financial Inclusion Prediction App')
st.write('Predict whether a person is likely to have a bank account.')

age = st.number_input('Age', min_value=0, max_value=120)
household_size = st.number_input('Household Size', min_value=1, max_value=20)

country = st.selectbox('Country', [0, 1, 2, 3])
gender = st.selectbox('Gender', [0, 1])

if st.button("Predict"):
    features = np.array([[age, household_size, country, gender]])
    prediction = model.predict(features)

    if prediction[0] == 1:
        st.success('✅ This person is likely to have a bank account.')
    else:
        st.error('❌ This person is not likely to have a bank account.')