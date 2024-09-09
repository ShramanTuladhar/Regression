import streamlit as st
import json
import pickle
import numpy as np
from PIL import Image

# Open an image file
image = Image.open(r"C:\Sydney_home_prices\app\sydney.jpeg")

# Display the image in the app
st.image(image,  width=500)

#st.title('SYDNEY SUBURB HOUSE PRICE PREDICTION')
st.markdown("<h1 style='text-align: center; color: green;'>SYDNEY SUBURB HOUSE PRICE PREDICTION</h1>", unsafe_allow_html=True)
# Specify the full path to the JSON file
json_file_path = r"C:\Sydney home prices\model\columns.json"

# Open the JSON file and load its contents
with open(json_file_path, 'r') as f:
    locations_data = json.load(f)

# Extract location names starting from column number 3
locations = locations_data['data_columns'][2:]

# Load the trained machine learning model
model_file_path = r"C:\Sydney home prices\model\Sydney home price model.pickle"
with open(model_file_path,'rb') as f :
    model= pickle.load(f)

def predict_price(bedrooms, bathrooms, location):
    # Find the index of the location in the 'locations' list
    loc_index = -1
    if location in locations:
        loc_index = locations.index(location) + 2  # Add 2 to account for the first two columns (bed and bath)
    else:
        print("Location not found in the 'locations' list.")
        return None

    # Preprocessing
    x_new = np.zeros(len(locations) + 2)  # Add 2 for bed and bath
    x_new[0] = bedrooms
    x_new[1] = bathrooms
    x_new[loc_index] = 1  # Set the location feature to 1

    # Predict the price using the model
    prediction = model.predict([x_new])[0]

    return prediction

# front end part
bedrooms = st.sidebar.slider('Bedrooms', 1, 10, 3)
bathrooms = st.sidebar.slider('Bathrooms', 1, 10, 2) 
location = st.sidebar.selectbox('Location', locations)

# Get prediction
predicted_price= None
if st.sidebar.button('Predict Price'):
    predicted_price = predict_price(bedrooms, bathrooms, location)

# Display prediction
if predicted_price is not None:
    #st.write(f'Predicted Price: ${predicted_price:.2f}')
    st.markdown(f"<h2 style='text-align: center; color: white;'>Predicted Price: ${predicted_price:.2f}</h2>", unsafe_allow_html=True)
    
