import streamlit as st
import pandas as pd
import joblib

model= joblib.load("mumbai_house_price_data_model.pkl")
encoder= joblib.load("mhp_label_encoder.pkl")

st.title("mumbai house price")

title= st.selectbox("title",encoder["title"].classes_)
price= st.number_input("price", 0,100000000)
area= st.number_input("area",0,10000)
price_per_sqft= st.number_input("price_per_sqft",0,10000)
locality= st.selectbox("locality",encoder["locality"].classes_)
city= st.selectbox("city",encoder["city"].classes_)
property_type= st.selectbox("property_type",encoder["property_type"].classes_)
bedroom_num= st.number_input("bedroom_num",0,10)
bathroom_num = st.number_input("bathroom_num", 0, 10)

balcony_num= st.number_input("balcony_num",0,10)
furnished= st.selectbox("furnished",encoder["furnished"].classes_)
age= st.number_input("age",0,100)
total_floors= st.number_input("total_floors",0,100)
latitude= st.number_input("latitude",0,100)
longitude= st.number_input("longitude",0,100)

df= pd.DataFrame({
    "title": [title],
    "price":[price],
    "area":[area],
    "price_per_sqft":[price_per_sqft],
    "locality":[locality],
    "city":[city],
    "property_type":[property_type],
    "bedroom_num":[bedroom_num],
    "bathroom_num":[bathroom_num],
    "balcony_num":[balcony_num],
    "furnished":[furnished],
    "age":[age],
    "total_floors":[total_floors],
    "latitude":[latitude],
    "longitude":[longitude]

})

if st.button("Predict"):
    for col in encoder:
        df[col] = encoder[col].transform(df[col])
        # Make sure columns are in same order as training
df = df[model.feature_names_in_]


prediction = model.predict(df)
st.success(f"Mumbai house price: {prediction[0]:,.2f}")
