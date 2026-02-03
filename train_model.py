"""
Training Script for Weather Prediction Model
This script trains the weather prediction model and saves it to disk
"""

import os
import joblib
from data_preprocessing import (
    generate_sample_data, 
    clean_data, 
    prepare_features, 
    split_data, 
    scale_features
)
from weather_predictor import WeatherPredictor
import config


def main():
    """
    Main training function
    """
    print("=" * 50)
    print("Weather Prediction Model Training")
    print("=" * 50)
    
    # Generate or load data
    print("\n1. Loading/Generating data...")
    data_file = os.path.join(config.DATA_DIR, 'sample_weather_data.csv')
    
    # Generate sample data if not exists
    if not os.path.exists(data_file):
        print("Generating sample weather data...")
        data = generate_sample_data(n_samples=1000)
        data.to_csv(data_file, index=False)
        print(f"Sample data saved to {data_file}")
    else:
        from data_preprocessing import load_data
        data = load_data(data_file)
    
    # Clean data
    print("\n2. Cleaning data...")
    cleaned_data = clean_data(data)
    
    # Prepare features
    print("\n3. Preparing features...")
    X, y = prepare_features(cleaned_data)
    
    # Split data
    print("\n4. Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Scale features
    print("\n5. Scaling features...")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Save scaler
    joblib.dump(scaler, config.SCALER_PATH)
    print(f"Scaler saved to {config.SCALER_PATH}")
    
    # Initialize and train model
    print("\n6. Training the model...")
    predictor = WeatherPredictor(n_estimators=100)
    predictor.train(X_train_scaled, y_train)
    
    # Evaluate model
    print("\n7. Evaluating the model...")
    metrics = predictor.evaluate(X_test_scaled, y_test)
    
    # Get feature importance
    print("\n8. Feature importance analysis...")
    predictor.get_feature_importance()
    
    # Save model
    print("\n9. Saving the trained model...")
    predictor.save_model()
    
    print("\n" + "=" * 50)
    print("Training completed successfully!")
    print(f"Model saved to: {config.MODEL_PATH}")
    print(f"Scaler saved to: {config.SCALER_PATH}")
    print(f"Model Accuracy: {metrics['accuracy']:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
