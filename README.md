# Weather Prediction ML Project

A machine learning application that predicts precipitation using Random Forest Regression based on weather parameters.

## Overview

This project implements a weather prediction system using Random Forest Regression to predict precipitation levels based on various weather features including temperature, humidity, atmospheric pressure, and wind speed.

## Features

- **Random Forest Regression Model**: Uses ensemble learning for accurate precipitation prediction
- **Interactive Prediction App**: User-friendly command-line interface for making predictions
- **Model Training Pipeline**: Complete training script with evaluation metrics and visualization
- **Sample Dataset**: Includes sample weather data for training and testing

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Aliakabr234234/Weather-predection-ML-project-.git
cd Weather-predection-ML-project-
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Train the Model

First, train the Random Forest model using the provided weather data:

```bash
python train_model.py
```

This will:
- Load and analyze the weather dataset
- Train a Random Forest Regression model
- Display evaluation metrics (RMSE, MAE, R² score)
- Show feature importance
- Generate a visualization plot (`prediction_results.png`)
- Save the trained model (`weather_model.pkl`)

### 2. Make Predictions

Use the trained model to predict precipitation:

```bash
python predict.py
```

The app will prompt you to enter:
- Temperature (°C)
- Humidity (%)
- Atmospheric Pressure (hPa)
- Wind Speed (km/h)

Example:
```
WEATHER PREDICTION APP
==========================================

Please enter the following weather parameters:
Temperature (°C): 25.5
Humidity (%): 70
Atmospheric Pressure (hPa): 1013.5
Wind Speed (km/h): 12.5

PREDICTION RESULTS
==========================================

Input Parameters:
  Temperature: 25.5°C
  Humidity: 70.0%
  Atmospheric Pressure: 1013.5 hPa
  Wind Speed: 12.5 km/h

Prediction:
  Expected Precipitation: 1.85 mm
  Weather Condition: Light precipitation expected
==========================================
```

## Model Details

- **Algorithm**: Random Forest Regression
- **Number of Estimators**: 100 trees
- **Max Depth**: 10
- **Features**: 
  - Temperature (°C)
  - Humidity (%)
  - Atmospheric Pressure (hPa)
  - Wind Speed (km/h)
- **Target**: Precipitation (mm)

## Project Structure

```
Weather-predection-ML-project-/
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
├── weather_data.csv          # Sample weather dataset
├── train_model.py            # Model training script
├── predict.py                # Prediction application
├── weather_model.pkl         # Trained model (generated after training)
└── prediction_results.png    # Visualization plot (generated after training)
```

## Model Performance

After training, the model provides:
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score
- Feature importance rankings

## Future Improvements

- Add more weather features (cloud cover, visibility, etc.)
- Implement additional ML algorithms (Gradient Boosting, XGBoost)
- Create a web-based interface using Flask or Streamlit
- Add real-time weather data fetching from APIs
- Implement time-series forecasting for multi-day predictions

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available for educational purposes.
