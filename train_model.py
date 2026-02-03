"""
Weather Prediction Model Training Script
Uses Random Forest Regression to predict precipitation based on weather features
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(file_path='weather_data.csv'):
    """Load weather data from CSV file"""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Data loaded successfully. Shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())
    print("\nData statistics:")
    print(df.describe())
    return df


def prepare_data(df):
    """Prepare features and target variable"""
    # Features: temperature, humidity, pressure, wind_speed
    # Target: precipitation
    X = df[['temperature', 'humidity', 'pressure', 'wind_speed']]
    y = df['precipitation']
    
    print("\nFeatures shape:", X.shape)
    print("Target shape:", y.shape)
    
    return X, y


def train_model(X_train, y_train):
    """Train Random Forest Regression model"""
    print("\nTraining Random Forest Regression model...")
    
    # Create and train the model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    print("Model trained successfully!")
    
    return model


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Evaluate model performance"""
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    # Training predictions
    y_train_pred = model.predict(X_train)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    # Testing predictions
    y_test_pred = model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print("\nTraining Set Metrics:")
    print(f"  RMSE: {train_rmse:.4f}")
    print(f"  MAE:  {train_mae:.4f}")
    print(f"  R2 Score: {train_r2:.4f}")
    
    print("\nTest Set Metrics:")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  MAE:  {test_mae:.4f}")
    print(f"  R2 Score: {test_r2:.4f}")
    
    # Feature importance
    print("\nFeature Importance:")
    feature_names = ['temperature', 'humidity', 'pressure', 'wind_speed']
    importances = model.feature_importances_
    for name, importance in zip(feature_names, importances):
        print(f"  {name}: {importance:.4f}")
    
    return y_test_pred


def plot_results(y_test, y_pred):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Precipitation')
    plt.ylabel('Predicted Precipitation')
    plt.title('Random Forest Regression: Actual vs Predicted Precipitation')
    plt.tight_layout()
    plt.savefig('prediction_results.png')
    print("\nPlot saved as 'prediction_results.png'")


def save_model(model, filename='weather_model.pkl'):
    """Save trained model to file"""
    joblib.dump(model, filename)
    print(f"\nModel saved as '{filename}'")


def main():
    """Main training pipeline"""
    print("="*50)
    print("WEATHER PREDICTION MODEL TRAINING")
    print("Using Random Forest Regression")
    print("="*50)
    
    # Load and prepare data
    df = load_data()
    X, y = prepare_data(df)
    
    # Split data into training and testing sets
    # Note: Using 20% test split. For production use, consider a larger dataset
    # or cross-validation for more robust evaluation.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    y_pred = evaluate_model(model, X_train, y_train, X_test, y_test)
    
    # Plot results
    plot_results(y_test, y_pred)
    
    # Save model
    save_model(model)
    
    print("\n" + "="*50)
    print("Training completed successfully!")
    print("="*50)


if __name__ == "__main__":
    main()
