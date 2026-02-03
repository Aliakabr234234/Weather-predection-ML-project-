"""
Configuration file for Weather Prediction ML Project
"""

import os

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

# Feature columns for weather prediction
FEATURE_COLUMNS = [
    'temperature',
    'humidity',
    'pressure',
    'wind_speed',
    'cloud_cover'
]

# Target column
TARGET_COLUMN = 'weather_condition'

# Weather conditions mapping
WEATHER_CONDITIONS = {
    0: 'Sunny',
    1: 'Cloudy',
    2: 'Rainy',
    3: 'Stormy'
}

# Model save path
MODEL_PATH = os.path.join(MODELS_DIR, 'weather_model.joblib')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.joblib')
