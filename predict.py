"""
Weather Prediction Application
Uses trained Random Forest model to predict precipitation
"""

import joblib
import numpy as np
import sys


def load_model(model_path='weather_model.pkl'):
    """Load trained model from file"""
    try:
        model = joblib.load(model_path)
        print("Model loaded successfully!")
        return model
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found.")
        print("Please train the model first by running: python train_model.py")
        sys.exit(1)


def get_user_input():
    """Get weather parameters from user"""
    print("\n" + "="*50)
    print("WEATHER PREDICTION APP")
    print("="*50)
    print("\nPlease enter the following weather parameters:")
    
    try:
        temperature = float(input("Temperature (°C): "))
        humidity = float(input("Humidity (%): "))
        pressure = float(input("Atmospheric Pressure (hPa): "))
        wind_speed = float(input("Wind Speed (km/h): "))
        
        return temperature, humidity, pressure, wind_speed
    except ValueError:
        print("Error: Please enter valid numeric values.")
        sys.exit(1)


def predict_precipitation(model, temperature, humidity, pressure, wind_speed):
    """Predict precipitation using the trained model"""
    # Prepare input data
    features = np.array([[temperature, humidity, pressure, wind_speed]])
    
    # Make prediction
    prediction = model.predict(features)[0]
    
    return prediction


def display_prediction(temperature, humidity, pressure, wind_speed, precipitation):
    """Display the prediction results"""
    print("\n" + "="*50)
    print("PREDICTION RESULTS")
    print("="*50)
    print("\nInput Parameters:")
    print(f"  Temperature: {temperature}°C")
    print(f"  Humidity: {humidity}%")
    print(f"  Atmospheric Pressure: {pressure} hPa")
    print(f"  Wind Speed: {wind_speed} km/h")
    print("\nPrediction:")
    print(f"  Expected Precipitation: {precipitation:.2f} mm")
    
    # Add interpretation
    if precipitation < 0.5:
        weather_condition = "No significant precipitation expected (Dry)"
    elif precipitation < 2.5:
        weather_condition = "Light precipitation expected"
    elif precipitation < 7.5:
        weather_condition = "Moderate precipitation expected"
    else:
        weather_condition = "Heavy precipitation expected"
    
    print(f"  Weather Condition: {weather_condition}")
    print("="*50)


def main():
    """Main prediction application"""
    # Load the trained model
    model = load_model()
    
    # Get user input
    temperature, humidity, pressure, wind_speed = get_user_input()
    
    # Make prediction
    precipitation = predict_precipitation(model, temperature, humidity, pressure, wind_speed)
    
    # Display results
    display_prediction(temperature, humidity, pressure, wind_speed, precipitation)


if __name__ == "__main__":
    main()
