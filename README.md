# Weather Prediction ML Project

A comprehensive Machine Learning project for predicting weather conditions based on meteorological features using Random Forest Classifier.

## 🌤️ Project Overview

This project implements a weather prediction system that uses machine learning to classify weather conditions into four categories:
- **Sunny**: Clear skies with minimal cloud cover
- **Cloudy**: Overcast conditions
- **Rainy**: Precipitation expected
- **Stormy**: Severe weather conditions

## 📋 Features

- **Data Preprocessing**: Automated data cleaning and feature scaling
- **Machine Learning Model**: Random Forest Classifier with optimized parameters
- **Prediction System**: Easy-to-use prediction interface for single or batch predictions
- **Model Evaluation**: Comprehensive performance metrics and feature importance analysis
- **Sample Data Generation**: Built-in sample data generator for testing

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/Aliakabr234234/Weather-predection-ML-project-.git
cd Weather-predection-ML-project-
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### Training the Model

Run the training script to train the weather prediction model:

```bash
python train_model.py
```

This will:
- Generate sample weather data (if not exists)
- Preprocess and clean the data
- Train a Random Forest model
- Evaluate model performance
- Save the trained model and scaler

### Making Predictions

After training, use the prediction script:

```bash
python predict.py
```

This will show example predictions for different weather scenarios.

### Using the Model Programmatically

```python
from predict import predict_single_sample

# Predict weather for specific conditions
result = predict_single_sample(
    temperature=28,      # Temperature in Celsius
    humidity=60,         # Humidity percentage
    pressure=1013,       # Atmospheric pressure in hPa
    wind_speed=12,       # Wind speed in km/h
    cloud_cover=40       # Cloud cover percentage
)

print(f"Predicted Weather: {result['weather']}")
print(f"Confidence: {result['confidence']:.2f}%")
```

## 📊 Model Features

The model uses the following meteorological features for prediction:

1. **Temperature** (°C): Ambient air temperature
2. **Humidity** (%): Relative humidity level
3. **Pressure** (hPa): Atmospheric pressure
4. **Wind Speed** (km/h): Wind velocity
5. **Cloud Cover** (%): Percentage of sky covered by clouds

## 📁 Project Structure

```
Weather-predection-ML-project-/
├── config.py                    # Configuration settings
├── data_preprocessing.py        # Data preprocessing functions
├── weather_predictor.py         # Weather prediction model class
├── train_model.py              # Model training script
├── predict.py                  # Prediction script
├── requirements.txt            # Project dependencies
├── README.md                   # Project documentation
├── .gitignore                  # Git ignore file
├── data/                       # Data directory
│   └── sample_weather_data.csv # Sample weather data
└── models/                     # Saved models directory
    ├── weather_model.joblib    # Trained model
    └── scaler.joblib          # Feature scaler
```

## 🔧 Configuration

Edit `config.py` to customize:
- Model parameters (random state, test size)
- Feature columns
- Weather condition mappings
- Model and data paths

## 📈 Model Performance

The model achieves high accuracy on the test set with detailed performance metrics including:
- Overall accuracy score
- Confusion matrix
- Precision, recall, and F1-score for each weather condition
- Feature importance rankings

## 🤝 Contributing

Contributions are welcome! Feel free to:
1. Fork the repository
2. Create a feature branch
3. Make your improvements
4. Submit a pull request

## 📝 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

Developed for demonstrating machine learning concepts in weather prediction.

## 🔮 Future Enhancements

- Add more weather features (precipitation amount, visibility, etc.)
- Implement deep learning models
- Create a web interface for predictions
- Add real-time weather data integration
- Improve model with ensemble methods
- Add time-series forecasting capabilities
