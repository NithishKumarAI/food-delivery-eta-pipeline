import streamlit as st
import pandas as pd

from src.predict import predict_eta

st.set_page_config(
    page_title="Food Delivery ETA Predictor",
    page_icon="🚚",
    layout="centered"
)

st.title("🚚 Food Delivery ETA Predictor")
st.info(
    "This is a demo Machine Learning project created for portfolio and educational purposes. "
    "Predictions are generated using a trained ML model and may not reflect real-world delivery systems."
)
st.markdown(
    "Predict food delivery time using a Machine Learning model"
)

st.write("---")

# --------------------------------
# Layout
# --------------------------------

col1, col2 = st.columns(2)

with col1:

    weather = st.selectbox(
        "Weather",
        ["Sunny", "Cloudy", "Fog", "Stormy", "Windy"]
    )

    vehicle = st.selectbox(
        "Vehicle Type",
        ["Bike", "Scooter", "Motorcycle"]
    )

    city = st.selectbox(
        "City Type",
        ["Urban", "Semi-Urban", "Metropolitan"]
    )

    age = st.slider(
        "Delivery Person Age",
        18,
        50,
        28
    )

    rating = st.slider(
        "Delivery Rating",
        1.0,
        5.0,
        4.5
    )

with col2:

    traffic = st.selectbox(
        "Traffic Density",
        ["Low", "Medium", "High", "Jam"]
    )

    order_type = st.selectbox(
        "Order Type",
        ["Snack", "Meal", "Buffet", "Drinks"]
    )

    festival = st.selectbox(
        "Festival",
        ["No", "Yes"]
    )

    multiple_deliveries = st.slider(
        "Multiple Deliveries",
        0,
        5,
        1
    )

    preparation_time = st.slider(
        "Preparation Time (mins)",
        5,
        60,
        20
    )

st.write("---")

# --------------------------------
# Predict Button
# --------------------------------

if st.button("Predict ETA"):

    try:

        sample_df = pd.DataFrame([{
            "Delivery_person_Age": age,
            "Delivery_person_Ratings": rating,

            "Restaurant_latitude": 12.9716,
            "Restaurant_longitude": 77.5946,

            "Delivery_location_latitude": 12.9352,
            "Delivery_location_longitude": 77.6245,

            "Weatherconditions": weather,
            "Road_traffic_density": traffic,

            "Vehicle_condition": 2,
            "Type_of_order": order_type,
            "Type_of_vehicle": vehicle,

            "multiple_deliveries": multiple_deliveries,
            "Festival": festival,
            "City": city,

            "Order_Date": "2024-05-26",
            "Time_Orderd": "18:45:00",
            "Time_Order_picked": "19:05:00",

            "preparation_time": preparation_time
        }])

        prediction = predict_eta(sample_df)

        st.success(
            f"Estimated Delivery Time: {prediction:.2f} minutes"
        )

    except Exception as e:

        st.error(f"Error: {e}")