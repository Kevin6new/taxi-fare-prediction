import streamlit as st
import requests
import joblib
import numpy as np
import re
from datetime import datetime
import folium
from streamlit_folium import folium_static
import polyline
from dotenv import load_dotenv
import os 

load_dotenv()

OPENCAGE_API_KEY = os.getenv("OPENCAGE_API_KEY")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

def get_google_directions_route(start_coords, end_coords, api_key):
    url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {
        "origin": f"{start_coords[0]},{start_coords[1]}",
        "destination": f"{end_coords[0]},{end_coords[1]}",
        "key": api_key
    }
    response = requests.get(url, params=params)
    directions = response.json()

    if directions['status'] == 'OK':
        polyline = directions['routes'][0]['overview_polyline']['points']
        return polyline
    else:
        return None

def draw_route(polyline_str, start_coords, end_coords):
    route_coordinates = polyline.decode(polyline_str)
    midpoint = [sum(x) / 2 for x in zip(start_coords, end_coords)]
    route_map = folium.Map(location=midpoint, zoom_start=14.25)

    folium.Marker(start_coords, icon=folium.Icon(color="green"), tooltip="Source").add_to(route_map)
    folium.Marker(end_coords, icon=folium.Icon(color="red"), tooltip="Destination").add_to(route_map)
    folium.PolyLine(route_coordinates, color='blue', weight=2.5, opacity=1).add_to(route_map)
    
    return route_map
    
def get_weather_data(vals, api_key):
    lat, lon = vals
    base_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=imperial"
    response = requests.get(base_url)
    weather_data = response.json()
    
    extracted_data = {
        'temperature': weather_data['main']['temp'],
        'feels_like': weather_data['main']['feels_like'],
        'humidity': weather_data['main']['humidity'],
        'wind_speed': weather_data['wind']['speed'],
        'visibility': weather_data.get('visibility', 0) / 1000.0,
        'precip_intensity': 0,
        'precip_probability': 0
    }
    if 'weather' in weather_data and len(weather_data['weather']) > 0:
        weather_data = weather_data['weather'][0]['description']
    
    description = weather_data.lower()

    # Mapping of weather keywords to short and long summaries
    short_summary_mapping = {
        'cloudy': ' Mostly Cloudy ',
        'rain': ' Rain ',
        'clear': ' Clear ',
        'partly cloudy': ' Partly Cloudy ',
        'overcast': ' Overcast ',
        'light rain': ' Light Rain ',
        'foggy': ' Foggy ',
        'drizzle': ' Drizzle '
    }

    long_summary_mapping = {
        'rain throughout the day': ' Rain throughout the day. ',
        'rain until morning': ' Rain until morning, starting again in the evening. ',
        'light rain in the morning': ' Light rain in the morning. ',
        'partly cloudy throughout the day': ' Partly cloudy throughout the day. ',
        'mostly cloudy throughout the day': ' Mostly cloudy throughout the day. ',
        'light rain overnight': ' Light rain in the morning and overnight. ',
        'light rain until evening': ' Light rain until evening. ',
        'foggy in the morning': ' Foggy in the morning. ',
        'overcast throughout the day': ' Overcast throughout the day. ',
        'possible drizzle in the morning': ' Possible drizzle in the morning. ',
        'rain in the morning and afternoon': ' Rain in the morning and afternoon. '
    }

    short_summary = next((summary for keyword, summary in short_summary_mapping.items() if keyword in description), ' Clear ')
    
    long_summary = next((summary for keyword, summary in long_summary_mapping.items() if keyword in description), ' Partly cloudy throughout the day. ')
    
    return extracted_data,short_summary, long_summary

def geocode_address(address, api_key):
    base_url = "https://api.opencagedata.com/geocode/v1/json"
    params = {'q': address, 'key': api_key}
    response = requests.get(base_url, params=params)
    data = response.json()
    if data['results']:
        lat = data['results'][0]['geometry']['lat']
        lon = data['results'][0]['geometry']['lng']
        return lat, lon
    return None, None

def calculate_road_distance(start_address, end_address, api_key):
    base_url = "https://maps.googleapis.com/maps/api/distancematrix/json"
    params = {
        'origins': start_address,
        'destinations': end_address,
        'units': 'imperial',
        'key': api_key,
    }
    response = requests.get(base_url, params=params)
    data = response.json()
    if data.get('rows') and data['rows'][0].get('elements'):
        element = data['rows'][0]['elements'][0]
        if element['status'] == 'OK':
            distance_text = element['distance']['text']
            return distance_text
        else:
            return "Error: " + element['status']
    else:
        return "Error: No data available."

# Load model and encoders
rf_model = joblib.load('xgb_model.joblib')
ordinal_encoder = joblib.load('ordinal_encoder.joblib')
one_hot_encoder = joblib.load('one_encoder.joblib')

# Streamlit UI
st.set_page_config(page_title="Ride Fare Prediction", page_icon="🚖", layout='wide')
st.title("Ride Fare Prediction")
st.info("🔧 This project is a work in progress and may be updated to enhance functionality and user experience.")
st.write("Please type the full address including the pin code and state.")

source_address = st.text_input("Enter your source address:")
destination_address = st.text_input("Enter your destination address:")

service_type = st.selectbox("Choose a service type:", ["Lyft", "Uber"])

if service_type == 'Lyft':
    cab_type_options = ["Shared", "Lux", "Lyft", "Lux Black XL", "Lyft XL", "Lux Black"]
else:  # service_type == 'Uber'
    cab_type_options = ["UberXL", "Black", "UberX", "WAV", "Black SUV", "UberPool", "Taxi"]

cab_type = st.selectbox("Choose a cab type:", cab_type_options)

service=one_hot_encoder.transform(np.array([[service_type]]))

if st.button("Predict Fare"):
    with st.spinner("Predicting the fare..."):
        source_coords = geocode_address(source_address, OPENCAGE_API_KEY)
        destination_coords = geocode_address(destination_address, OPENCAGE_API_KEY)
        
        if source_coords and destination_coords:
            distance = calculate_road_distance(source_address, destination_address, GOOGLE_MAPS_API_KEY)
            if source_coords and destination_coords:
                st.error("Source and destination address cannot be the same")
            if "Error" not in distance:
                # Extract only the numerical part from distance
                distance_value = float(re.findall(r"\d+\.\d+", distance)[0])
                
                current_weather,short,longer = get_weather_data(source_coords, OPENWEATHER_API_KEY)
                categorical_data = np.array([[longer, short,cab_type]])
                encoded_categorical = ordinal_encoder.transform(categorical_data)
                # Prepare features

                current_datetime = datetime.now()
                features = [
                    current_datetime.hour, current_datetime.day, current_datetime.month,
                    distance_value, 1.0,  # Assume surge multiplier is 1.0
                    current_weather['temperature'], current_weather['feels_like'],
                    0, 0,  # precipitation data
                    current_weather['humidity'], current_weather['wind_speed'], current_weather['visibility'],
                    # Encode categorical data
                    encoded_categorical[0, 0],  # Encoded long_summary
                    encoded_categorical[0, 1],  # Encoded short_summary
                    encoded_categorical[0, 2],    # name (assuming numerical encoding or placeholder)
                    service[0, 0],    # cab_type_Lyft (binary)
                    service[0, 1]     # cab_type_Uber (binary)
                ]
                # Make prediction
                
                predicted_fare = rf_model.predict(np.array(features).reshape(1, -1))
                st.success(f"Predicted Fare: ${predicted_fare[0]:.2f} for a distance of {distance}")
                route_polyline = get_google_directions_route(source_coords, destination_coords, GOOGLE_MAPS_API_KEY)
                if route_polyline:
                    route_map = draw_route(route_polyline, source_coords, destination_coords)
                    folium_static(route_map)
                else:
                    st.error("Failed to retrieve the route. Please check your API key and quota.")
            else:
                st.error(distance)
        else:
            st.error("Failed to geocode one or both addresses.")

