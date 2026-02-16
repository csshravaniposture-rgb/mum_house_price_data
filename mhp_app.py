import streamlit as st
import pandas as pd
import joblib

# Load model files
model = joblib.load("mumbai_house_price_data_model.pkl")
encoder = joblib.load("mhp_label_encoder.pkl")

st.title("Mumbai House Price Prediction")

title = st.selectbox("Title", encoder["title"].classes_)
price = st.number_input("Price", 0, 100000000)
area = st.number_input("Area", 0, 10000)
price_per_sqft = st.number_input("Price per Sqft", 0, 100000)
locality = st.selectbox("Locality", encoder["locality"].classes_)
city = st.selectbox("City", encoder["city"].classes_)
property_type = st.selectbox("Property Type", encoder["property_type"].classes_)
bedroom_num = st.number_input("Bedroom Number", 0, 10)
bathroom_num = st.number_input("Bathroom Number", 0, 10)
balcony_num = st.number_input("Balcony Number", 0, 10)
furnished = st.selectbox("Furnished", encoder["furnished"].classes_)
age = st.number_input("Age", 0, 100)
total_floors = st.number_input("Total Floors", 0, 100)
latitude = st.number_input("Latitude", 0.0, 100.0)
longitude = st.number_input("Longitude", 0.0, 100.0)

df = pd.DataFrame({
    "title": [title],
    "price": [price],
    "area": [area],
    "price_per_sqft": [price_per_sqft],
    "locality": [locality],
    "city": [city],
    "property_type": [property_type],
    "bedroom_num": [bedroom_num],
    "bathroom_num": [bathroom_num],
    "balcony_num": [balcony_num],
    "furnished": [furnished],
    "age": [age],
    "total_floors": [total_floors],
    "latitude": [latitude],
    "longitude": [longitude]
})

if st.button("Predict"):
    for col in encoder:
        df[col] = encoder[col].transform(df[col])

    prediction = model.predict(df)
    st.success(f"Mumbai House Price: ₹ {prediction[0]:,.2f}")
