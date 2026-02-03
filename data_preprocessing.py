"""
Data Preprocessing Module for Weather Prediction
This module handles data loading, cleaning, and preprocessing
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import config


def load_data(file_path):
    """
    Load weather data from CSV file
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded data
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully. Shape: {data.shape}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def clean_data(data):
    """
    Clean the weather data by handling missing values and outliers
    
    Args:
        data (pd.DataFrame): Raw data
        
    Returns:
        pd.DataFrame: Cleaned data
    """
    # Make a copy to avoid modifying original data
    cleaned_data = data.copy()
    
    # Handle missing values
    print(f"Missing values before cleaning:\n{cleaned_data.isnull().sum()}")
    
    # Fill missing values with median for numerical columns
    for col in config.FEATURE_COLUMNS:
        if col in cleaned_data.columns:
            if cleaned_data[col].isnull().sum() > 0:
                cleaned_data[col].fillna(cleaned_data[col].median(), inplace=True)
    
    # Remove rows with missing target values
    if config.TARGET_COLUMN in cleaned_data.columns:
        cleaned_data = cleaned_data.dropna(subset=[config.TARGET_COLUMN])
    
    print(f"Missing values after cleaning:\n{cleaned_data.isnull().sum()}")
    print(f"Data shape after cleaning: {cleaned_data.shape}")
    
    return cleaned_data


def prepare_features(data):
    """
    Prepare features and target variables
    
    Args:
        data (pd.DataFrame): Cleaned data
        
    Returns:
        tuple: (X, y) features and target
    """
    # Extract features
    X = data[config.FEATURE_COLUMNS].values
    
    # Extract target
    y = data[config.TARGET_COLUMN].values
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    return X, y


def split_data(X, y, test_size=None, random_state=None):
    """
    Split data into training and testing sets
    
    Args:
        X (array): Features
        y (array): Target
        test_size (float): Proportion of test set
        random_state (int): Random seed
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    if test_size is None:
        test_size = config.TEST_SIZE
    if random_state is None:
        random_state = config.RANDOM_STATE
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """
    Scale features using StandardScaler
    
    Args:
        X_train (array): Training features
        X_test (array): Test features
        
    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Features scaled successfully")
    
    return X_train_scaled, X_test_scaled, scaler


def generate_sample_data(n_samples=1000):
    """
    Generate sample weather data for demonstration
    
    Args:
        n_samples (int): Number of samples to generate
        
    Returns:
        pd.DataFrame: Generated sample data
    """
    np.random.seed(config.RANDOM_STATE)
    
    # Generate features
    temperature = np.random.uniform(0, 40, n_samples)
    humidity = np.random.uniform(20, 100, n_samples)
    pressure = np.random.uniform(980, 1040, n_samples)
    wind_speed = np.random.uniform(0, 30, n_samples)
    cloud_cover = np.random.uniform(0, 100, n_samples)
    
    # Generate weather conditions based on features
    weather_condition = []
    for i in range(n_samples):
        if temperature[i] > 25 and humidity[i] < 50 and cloud_cover[i] < 30:
            weather_condition.append(0)  # Sunny
        elif humidity[i] > 80 and cloud_cover[i] > 70:
            if wind_speed[i] > 20:
                weather_condition.append(3)  # Stormy
            else:
                weather_condition.append(2)  # Rainy
        elif cloud_cover[i] > 50:
            weather_condition.append(1)  # Cloudy
        else:
            # Random assignment for edge cases
            weather_condition.append(np.random.randint(0, 4))
    
    # Create DataFrame
    data = pd.DataFrame({
        'temperature': temperature,
        'humidity': humidity,
        'pressure': pressure,
        'wind_speed': wind_speed,
        'cloud_cover': cloud_cover,
        'weather_condition': weather_condition
    })
    
    return data
