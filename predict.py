"""
Prediction Script for Weather Prediction Model
This script loads a trained model and makes predictions
"""

import numpy as np
import joblib
from weather_predictor import WeatherPredictor
import config


def predict_single_sample(temperature, humidity, pressure, wind_speed, cloud_cover):
    """
    Predict weather for a single sample
    
    Args:
        temperature (float): Temperature in Celsius
        humidity (float): Humidity percentage
        pressure (float): Atmospheric pressure in hPa
        wind_speed (float): Wind speed in km/h
        cloud_cover (float): Cloud cover percentage
        
    Returns:
        dict: Prediction results
    """
    # Load the trained model and scaler
    try:
        predictor = WeatherPredictor()
        predictor.load_model()
        scaler = joblib.load(config.SCALER_PATH)
    except FileNotFoundError:
        print("Error: Trained model not found. Please run train_model.py first.")
        return None
    
    # Prepare input data
    input_data = np.array([[temperature, humidity, pressure, wind_speed, cloud_cover]])
    
    # Scale the input
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    results = predictor.predict_weather_description(input_scaled)
    
    return results[0]


def predict_batch(data):
    """
    Predict weather for multiple samples
    
    Args:
        data (array): Array of features [n_samples, n_features]
        
    Returns:
        list: Prediction results for all samples
    """
    # Load the trained model and scaler
    try:
        predictor = WeatherPredictor()
        predictor.load_model()
        scaler = joblib.load(config.SCALER_PATH)
    except FileNotFoundError:
        print("Error: Trained model not found. Please run train_model.py first.")
        return None
    
    # Scale the input
    data_scaled = scaler.transform(data)
    
    # Make predictions
    results = predictor.predict_weather_description(data_scaled)
    
    return results


def main():
    """
    Main prediction function with example usage
    """
    print("=" * 50)
    print("Weather Prediction System")
    print("=" * 50)
    
    # Example predictions
    print("\nExample 1: Sunny Day")
    result1 = predict_single_sample(
        temperature=30,
        humidity=40,
        pressure=1015,
        wind_speed=10,
        cloud_cover=20
    )
    if result1:
        print(f"Predicted Weather: {result1['weather']}")
        print(f"Confidence: {result1['confidence']:.2f}%")
        print("Probabilities for all conditions:")
        for condition, prob in result1['probabilities'].items():
            print(f"  {condition}: {prob:.2f}%")
    
    print("\n" + "-" * 50)
    
    print("\nExample 2: Rainy Day")
    result2 = predict_single_sample(
        temperature=18,
        humidity=85,
        pressure=1005,
        wind_speed=15,
        cloud_cover=90
    )
    if result2:
        print(f"Predicted Weather: {result2['weather']}")
        print(f"Confidence: {result2['confidence']:.2f}%")
        print("Probabilities for all conditions:")
        for condition, prob in result2['probabilities'].items():
            print(f"  {condition}: {prob:.2f}%")
    
    print("\n" + "-" * 50)
    
    print("\nExample 3: Stormy Day")
    result3 = predict_single_sample(
        temperature=15,
        humidity=90,
        pressure=990,
        wind_speed=25,
        cloud_cover=95
    )
    if result3:
        print(f"Predicted Weather: {result3['weather']}")
        print(f"Confidence: {result3['confidence']:.2f}%")
        print("Probabilities for all conditions:")
        for condition, prob in result3['probabilities'].items():
            print(f"  {condition}: {prob:.2f}%")
    
    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
