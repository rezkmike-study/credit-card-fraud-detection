import os
import streamlit as st
import pandas as pd
import joblib 
import numpy as np
from sklearn.preprocessing import StandardScaler

# Paths for the model, scaler, PCA, and encoders
model_path = '/mount/src/credit-card-fraud-detection/fraud/fraud_detection_model.joblib'
scaler_path = '/mount/src/credit-card-fraud-detection/fraud/scaler.joblib'
pca_path = '/mount/src/credit-card-fraud-detection/fraud/pca.joblib'
onehot_encoder_path = '/mount/src/credit-card-fraud-detection/fraud/onehot_encoder.joblib'

# Load objects if they exist, else stop the app
if not all(map(os.path.isfile, [model_path, scaler_path, pca_path, onehot_encoder_path])):
    st.error("Error: Necessary model files are missing. meow1")
    st.stop()

# model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
pca = joblib.load(pca_path)
onehot_encoder = joblib.load(onehot_encoder_path)

def preprocess_input(input_data):
    # Ensure the input_data columns match the expected training columns
    training_column_order = ['amt', 'lat', 'long', 'merch_lat', 'merch_long', 'city_pop', 
                             'category_shopping_pos', 'category_entertainment', 'category_gas_transport', 
                             'category_grocery_net', 'category_grocery_pos', 'category_misc_net', 'category_misc_pos',
                             'category_shopping_net', 'city_Orient', 'city_Malad', 'city_City', 'city_Grenada', 'city_High', 
                             'city_Rolls', 'city_Mountain', 'city_Park', 'city_Freedom', 'city_Honokaa', 'city_Valentine', 
                             'city_Westfir', 'city_Thompson', 'city_Conway', 'city_Athena', 'city_San', 'city_Jose', 
                             'city_Ravenna', 'city_Parks', 'city_Fort', 'city_Washakie', 'city_Littleton', 
                             'city_Meadville', 'city_Moab', 'city_Hawthorne', 'city_Manville', 'city_June', 
                             'city_Lake', 'city_Sixes', 'city_Holstein', 'city_Westerville', 'city_Ballwin', 
                             'city_Fields', 'city_Landing', 'city_Louisiana', 'city_Kansas', 'city_City', 'city_Mesa', 
                             'city_Lonetree', 'city_Centerview', 'city_Colorado', 'city_Springs', 'city_Blairsden-Graeagle', 
                             'city_Cardwell', 'city_Phoenix', 'city_Newhall', 'city_Tomales', 'city_Redford', 'city_Weeping', 
                             'city_Water', 'city_Portland', 'city_Iliff', 'city_Burlington', 'city_Wales', 'city_Mound', 'city_City', 
                             'city_Greenview', 'city_Lakeport', 'city_Llano', 'city_Carlotta', 'city_Dumont', 'city_Fullerton', 
                             'city_North', 'city_Loup', 'city_Browning', 'city_Kent', 'city_Fiddletown', 'city_Huntington', 
                             'city_Beach', 'city_Meridian', 'city_Glendale', 'city_Alva', 'city_Blairstown', 'city_Laguna', 
                             'city_Hills', 'city_Albuquerque', 'city_Azusa', 'city_Gardiner', 'city_Rock', 'city_Springs', 
                             'city_Paauilo', 'city_Eugene', 'city_Daly', 'city_City', 'city_Mendon', 'city_Powell', 'city_Butte',  'state_OR']

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
    lat = st.sidebar.number_input('Transaction Latitude', value=43, min_value=-90, max_value=90, step=1000)
    long = st.sidebar.number_input('Transaction Longitude', value=-108, min_value=-180, max_value=180, step=1000)
    merch_lat = st.sidebar.number_input('Merchant Latitude', value=43, min_value=-90, max_value=90, step=1000)
    merch_long = st.sidebar.number_input('Merchant Longitude', value=-108, min_value=-180, max_value=180, step=1000)
    city_pop = st.sidebar.number_input('City Population', value=50000, min_value=10000, max_value=1000000, step=1000)

    # Categorical features - Replace with actual categories from your dataset
    category = st.sidebar.selectbox('Transaction Category', ('shopping_pos', 'entertainment', 'gas_transport', 'grocery_net', 'grocery_pos', 'misc_net'))
    city = st.sidebar.selectbox('City', ('Sixes'))
    state = st.sidebar.selectbox('State', ('OR'))

    # Combine the features into a dataframe
    data = {
        'amt': amt,
        'lat': lat,
        'long': long,
        'merch_lat': merch_lat,
        'merch_long': merch_long,
        'city_pop': city_pop,
        'category_shopping_pos': 1 if category == 'shopping_pos' else 0,
        'category_entertainment': 1 if category == 'entertainment' else 0,
        'category_gas_transport': 1 if category == 'gas_transport' else 0,
        'category_grocery_net': 1 if category == 'grocery_net' else 0,
        'category_grocery_pos': 1 if category == 'grocery_pos' else 0,
        'category_misc_net': 1 if category == 'misc_net' else 0,
        'category_misc_pos': 1 if category == 'misc_pos' else 0,
        'category_shopping_net': 1 if category == 'shopping_net' else 0,
        'city_Orient': 1 if city == 'Orient' else 0,
        'city_Malad': 1 if city == 'Malad' else 0,
        'city_City': 1 if city == 'City' else 0,
        'city_Grenada': 1 if city == 'Grenada' else 0,
        'city_High': 1 if city == 'High' else 0,
        'city_Rolls': 1 if city == 'Rolls' else 0,
        'city_Mountain': 1 if city == 'Mountain' else 0,
        'city_Park': 1 if city == 'Park' else 0,
        'city_Freedom': 1 if city == 'Freedom' else 0,
        'city_Honokaa': 1 if city == 'Honokaa' else 0,
        'city_Valentine': 1 if city == 'Valentine' else 0,
        'city_Westfir': 1 if city == 'Westfir' else 0,
        'city_Thompson': 1 if city == 'Thompson' else 0,
        'city_Conway': 1 if city == 'Conway' else 0,
        'city_Athena': 1 if city == 'Athena' else 0,
        'city_San': 1 if city == 'San' else 0,
        'city_Jose': 1 if city == 'Jose' else 0,
        'city_Ravenna': 1 if city == 'Ravenna' else 0,
        'city_Parks': 1 if city == 'Parks' else 0,
        'city_Fort': 1 if city == 'Fort' else 0,
        'city_Washakie': 1 if city == 'Washakie' else 0,
        'city_Littleton': 1 if city == 'Littleton' else 0,
        'city_Meadville': 1 if city == 'Meadville' else 0,
        'city_Moab': 1 if city == 'Moab' else 0,
        'city_Hawthorne': 1 if city == 'Hawthorne' else 0,
        'city_Manville': 1 if city == 'Manville' else 0,
        'city_June': 1 if city == 'June' else 0,
        'city_Lake': 1 if city == 'Lake' else 0,
        'city_Sixes': 1 if city == 'Sixes' else 0,
        'city_Holstein': 1 if city == 'Holstein' else 0,
        'city_Westerville': 1 if city == 'Westerville' else 0,
        'city_Ballwin': 1 if city == 'Ballwin' else 0,
        'city_Fields': 1 if city == 'Fields' else 0,
        'city_Landing': 1 if city == 'Landing' else 0,
        'city_Louisiana': 1 if city == 'Louisiana' else 0,
        'city_Kansas': 1 if city == 'Kansas' else 0,
        'city_City': 1 if city == 'City' else 0,
        'city_Mesa': 1 if city == 'Mesa' else 0,
        'city_Lonetree': 1 if city == 'Lonetree' else 0,
        'city_Centerview': 1 if city == 'Centerview' else 0,
        'city_Colorado': 1 if city == 'Colorado' else 0,
        'city_Springs': 1 if city == 'Springs' else 0,
        'city_Blairsden-Graeagle': 1 if city == 'Blairsden-Graeagle' else 0,
        'city_Cardwell': 1 if city == 'Cardwell' else 0,
        'city_Phoenix': 1 if city == 'Phoenix' else 0,
        'city_Newhall': 1 if city == 'Newhall' else 0,
        'city_Tomales': 1 if city == 'Tomales' else 0,
        'city_Redford': 1 if city == 'Redford' else 0,
        'city_Weeping': 1 if city == 'Weeping' else 0,
        'city_Water': 1 if city == 'Water' else 0,
        'city_Portland': 1 if city == 'Portland' else 0,
        'city_Iliff': 1 if city == 'Iliff' else 0,
        'city_Burlington': 1 if city == 'Burlington' else 0,
        'city_Wales': 1 if city == 'Wales' else 0,
        'city_Mound': 1 if city == 'Mound' else 0,
        'city_City': 1 if city == 'City' else 0,
        'city_Greenview': 1 if city == 'Greenview' else 0,
        'city_Lakeport': 1 if city == 'Lakeport' else 0,
        'city_Llano': 1 if city == 'Llano' else 0,
        'city_Carlotta': 1 if city == 'Carlotta' else 0,
        'city_Dumont': 1 if city == 'Dumont' else 0,
        'city_Fullerton': 1 if city == 'Fullerton' else 0,
        'city_North': 1 if city == 'North' else 0,
        'city_Loup': 1 if city == 'Loup' else 0,
        'city_Browning': 1 if city == 'Browning' else 0,
        'city_Kent': 1 if city == 'Kent' else 0,
        'city_Fiddletown': 1 if city == 'Fiddletown' else 0,
        'city_Huntington': 1 if city == 'Huntington' else 0,
        'city_Beach': 1 if city == 'Beach' else 0,
        'city_Meridian': 1 if city == 'Meridian' else 0,
        'city_Glendale': 1 if city == 'Glendale' else 0,
        'city_Alva': 1 if city == 'Alva' else 0,
        'city_Blairstown': 1 if city == 'Blairstown' else 0,
        'city_Laguna': 1 if city == 'Laguna' else 0,
        'city_Hills': 1 if city == 'Hills' else 0,
        'city_Albuquerque': 1 if city == 'Albuquerque' else 0,
        'city_Azusa': 1 if city == 'Azusa' else 0,
        'city_Gardiner': 1 if city == 'Gardiner' else 0,
        'city_Rock': 1 if city == 'Rock' else 0,
        'city_Springs': 1 if city == 'Springs' else 0,
        'city_Paauilo': 1 if city == 'Paauilo' else 0,
        'city_Eugene': 1 if city == 'Eugene' else 0,
        'city_Daly': 1 if city == 'Daly' else 0,
        'city_City': 1 if city == 'City' else 0,
        'city_Mendon': 1 if city == 'Mendon' else 0,
        'city_Powell': 1 if city == 'Powell' else 0,
        'city_Butte': 1 if city == 'Butte' else 0,
        'state_OR': 1 if state == 'OR' else 0,
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
