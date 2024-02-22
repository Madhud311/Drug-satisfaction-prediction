#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor

html_temp = """
<style>
    body {
        background-image: url('background.jpg');
        background-size: cover;
    }
</style>
<div class="content">
    <h1 style="color: black; text-align: center;">Drug Performance Evaluation</h1>
    <br>
"""
st.markdown(html_temp, unsafe_allow_html=True)

#st.sidebar.header('User Input Parameters')

df1 = pd.read_csv('Drug.csv')

def user_input_features():
    condition = st.selectbox('Condition', df1['Condition'].unique(), key='condition_selectbox')
    drug = st.selectbox('Drug', df1['Drug'].unique(), key='drug_selectbox')
    indication = st.selectbox('Indication', df1['Indication'].unique(), key='indication_selectbox')
    type_val = st.selectbox('Type', df1['Type'].unique(), key='type_selectbox')
    reviews = st.slider('Reviews', min_value=0, max_value=994, value=36, key='reviews_slider')
    effective = st.slider('Effective', min_value=0.0, max_value=5.0, value=5.0, key='effective_slider')
    ease_of_use = st.slider('EaseOfUse', min_value=0.0, max_value=5.0, value=5.0, key='ease_of_use_slider')
    
    le = LabelEncoder()
    condition = le.fit_transform([condition])[0]
    drug = le.fit_transform([drug])[0]
    indication = le.fit_transform([indication])[0]
    type_val = le.fit_transform([type_val])[0]

    data = {
        'Condition': condition,
        'Drug': drug,
        'Indication': indication,
        'Type': type_val,
        'Reviews': reviews,
        'Effective': effective,
        'EaseOfUse': ease_of_use
    }
    features = pd.DataFrame(data, index=[0])

    return features

df=user_input_features()


X = df1.drop('Satisfaction', axis=1)
Y = df1['Satisfaction']

# Split the data into training
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Extract categorical columns
categorical_cols = ['Condition', 'Drug', 'Indication', 'Type']

# Label encode categorical columns in the training set
for col in categorical_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])

# Label encode categorical columns in the user input
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform([df[col][0]])  # Use [0] as LabelEncoder expects a sequence

# Standardize the data
scaler = StandardScaler()

# Fit the scaler on the training set and transform the user input
X_train_scaled = scaler.fit_transform(X_train)
user_input_scaled = scaler.transform(df)

gradient_boosting_regressor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gradient_boosting_regressor.fit(X_train_scaled, y_train)

# Predict the satisfaction value for the user input parameters
y_pred_user_input = gradient_boosting_regressor.predict(user_input_scaled)

st.subheader('Satisfaction Prediction for User Input Parameters')
st.write(y_pred_user_input[0])


# In[ ]:




