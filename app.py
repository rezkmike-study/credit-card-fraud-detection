import os
import streamlit as st
import pandas as pd
import joblib 
import numpy as np
from sklearn.preprocessing import StandardScaler

# Paths for the model, scaler, PCA, and encoders
model_path = 'fraud.joblib'
scaler_path = 'scaler.joblib'
pca_path = 'pca.joblib'
onehot_encoder_path = 'onehot_encoder.joblib'

# Load objects if they exist, else stop the app
if not all(map(os.path.isfile, [model_path, scaler_path, pca_path, onehot_encoder_path])):
    st.error("Error: Necessary model files are missing.")
    st.stop()

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
pca = joblib.load(pca_path)
onehot_encoder = joblib.load(onehot_encoder_path)

def preprocess_input(input_data):
    # Ensure the input_data columns match the expected training columns
    training_column_order = ['amt', 'lat', 'long',
                             'merch_lat', 'merch_long', 'city_pop',
                             'merchant_Cummerata-Jones', 'merchant_Pouros-Haag', 'merchant_Larson-Moen',
                             'merchant_Goodwin-Nitzsche', 'city_Sixes', 'city_Riverton',
                             'city_Superior', 'city_Manley', 'state_OR',
                             'state_WY', 'state_OR', 'state_NE',
                             'job', 'dob', 'trans_num', 'trans_date_trans_time']

    # Reorder the columns in the input_data DataFrame to match the training order
    input_data = input_data[training_column_order]

    # Scale the features using the loaded scaler
    scaled_features = scaler.transform(input_data)

    # Apply PCA transformation
    pca_features = pca.transform(scaled_features)

    return pca_features


def user_input_features():
    st.sidebar.header("User Input Features")

    # Numerical features - Adjust the ranges and default values as needed
    amt = st.sidebar.number_input('Transaction Amount', value=50000, min_value=10000, max_value=1000000, step=1000)
    lat = st.sidebar.number_input('Transaction Latitude', value=43.0351, min_value=-90, max_value=90, step=1000)
    long = st.sidebar.number_input('Transaction Longitude', value=-108.2024, min_value=-180, max_value=180, step=1000)
    merch_lat = st.sidebar.number_input('Merchant Latitude', value=43.0351, min_value=-90, max_value=90, step=1000)
    merch_long = st.sidebar.number_input('Merchant Longitude', value=-108.2024, min_value=-180, max_value=180, step=1000)
    city_pop = st.sidebar.slider('City Population', 0.0, 1000000, 50000)

    # Categorical features - Replace with actual categories from your dataset
    merchant = st.sidebar.selectbox('Merchant', ('Cummerata-Jones', 'Pouros-Haag', 'Larson-Moen', 'Goodwin-Nitzsche'))
    category = st.sidebar.selectbox('Transaction Category', ('shopping_pos'))
    city = st.sidebar.selectbox('City', ('Sixes', 'Riverton', 'Superior', 'Manley'))
    state = st.sidebar.selectbox('State', ('OR', 'WY', 'AZ', 'NE'))
    job = st.sidebar.selectbox('Job', ('Video editor'))
    dob = st.sidebar.selectbox('DOB', ('14/09/1956'))
    trans_num = st.sidebar.selectbox('Transaction Number', ('b2cd7d04dbb330dd548105b12ak44ba4'))
    trans_date_trans_time = st.sidebar.selectbox('Transaction Number', ('16/01/2019 11:26'))

    # Combine the features into a dataframe
    data = {
        'amt': amt,
        'lat': lat,
        'long': long,
        'merch_lat': merch_lat,
        'merch_long': merch_long,
        'city_pop': city_pop,
        'merchant_Cummerata-Jones': 1 if merchant == 'Cummerata-Jones' else 0,
        'merchant_Pouros-Haag': 1 if merchant == 'Pouros-Haag' else 0,
        'merchant_Larson-Moen': 1 if merchant == 'Larson-Moen' else 0,
        'merchant_Goodwin-Nitzsche': 1 if merchant == 'Goodwin-Nitzsche' else 0,
        'category_': 1 if category == 'shopping_pos' else 0,
        'city_Sixes': 1 if city == 'Sixes' else 0,
        'city_Riverton': 1 if city == 'Riverton' else 0,
        'city_Superior': 1 if city == 'Superior' else 0,
        'city_Manley': 1 if city == 'Manley' else 0,
        'state_OR': 1 if state == 'OR' else 0,
        'state_WY': 1 if state == 'WY' else 0,
        'state_OR': 1 if state == 'AZ' else 0,
        'state_NE': 1 if state == 'NE' else 0,
        'job': 1 if job == 'Video editor' else 0,
        'dob': 1 if dob == '14/09/1956' else 0,
        'trans_num': 1 if trans_num == 'b2cd7d04dbb330dd548105b12ak44ba4' else 0,
        'trans_date_trans_time': 1 if trans_date_trans_time == '16/01/2019 11:26' else 0
    }

    features = pd.DataFrame(data, index=[0])
    return features


def main():
    st.title("Fraud Transaction Prediction")

    # Custom Styling with a border for the description
    st.markdown("""
        <style>
        .main {
            background-color: #F5F5F5;
        }
        .description {
            border: 1px solid #4F8BF9;
            border-radius: 5px;
            padding: 10px;
            margin-bottom: 20px;
        }
        </style>
        """, unsafe_allow_html=True)

    # Introduction text with border
    st.markdown("""
    <div class="description">
        <p>Welcome to the Fraud Prediction Application. This tool is designed to help you 
        understand the likelihood of a loan being approved based on various factors such as 
        amount, merchant, city, and more. Simply adjust the parameters in the sidebar 
        to match your details and click 'Predict' to see the outcome.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        input_df = user_input_features()

    # Button to make prediction
    if st.sidebar.button('Predict'):
        # Preprocess input data
        preprocessed_input = preprocess_input(input_df)
        
        # Make prediction
        prediction = model.predict(preprocessed_input)
        
        # Convert prediction to interpretable output
        prediction_text = "Approved" if prediction[0] == 1 else "Denied"

        # Display the prediction with styling
        st.subheader("Prediction")
        st.write(f"The transaction status is: **{prediction_text}**")


if __name__ == '__main__':
    main()
