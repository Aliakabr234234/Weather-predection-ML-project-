"""
Weather Prediction Model
This module contains the machine learning model for weather prediction
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import config


class WeatherPredictor:
    """
    Weather Prediction Model using Random Forest Classifier
    """
    
    def __init__(self, n_estimators=100, random_state=None):
        """
        Initialize the Weather Predictor
        
        Args:
            n_estimators (int): Number of trees in the forest
            random_state (int): Random seed for reproducibility
        """
        if random_state is None:
            random_state = config.RANDOM_STATE
        
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        self.scaler = None
        self.is_trained = False
    
    def train(self, X_train, y_train):
        """
        Train the weather prediction model
        
        Args:
            X_train (array): Training features
            y_train (array): Training labels
        """
        print("Training the weather prediction model...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print("Model training completed!")
    
    def predict(self, X):
        """
        Make predictions on new data
        
        Args:
            X (array): Features to predict
            
        Returns:
            array: Predicted weather conditions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = self.model.predict(X)
        return predictions
    
    def predict_proba(self, X):
        """
        Get prediction probabilities
        
        Args:
            X (array): Features to predict
            
        Returns:
            array: Prediction probabilities for each class
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        probabilities = self.model.predict_proba(X)
        return probabilities
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model performance
        
        Args:
            X_test (array): Test features
            y_test (array): Test labels
            
        Returns:
            dict: Evaluation metrics
        """
        predictions = self.predict(X_test)
        
        accuracy = accuracy_score(y_test, predictions)
        conf_matrix = confusion_matrix(y_test, predictions)
        class_report = classification_report(
            y_test, 
            predictions,
            target_names=[config.WEATHER_CONDITIONS[i] for i in range(len(config.WEATHER_CONDITIONS))]
        )
        
        print(f"\nModel Accuracy: {accuracy:.4f}")
        print("\nConfusion Matrix:")
        print(conf_matrix)
        print("\nClassification Report:")
        print(class_report)
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report
        }
    
    def get_feature_importance(self):
        """
        Get feature importance scores
        
        Returns:
            dict: Feature importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        importance = self.model.feature_importances_
        feature_importance = dict(zip(config.FEATURE_COLUMNS, importance))
        
        print("\nFeature Importance:")
        for feature, score in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
            print(f"{feature}: {score:.4f}")
        
        return feature_importance
    
    def save_model(self, model_path=None):
        """
        Save the trained model to disk
        
        Args:
            model_path (str): Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        if model_path is None:
            model_path = config.MODEL_PATH
        
        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path=None):
        """
        Load a trained model from disk
        
        Args:
            model_path (str): Path to load the model from
        """
        if model_path is None:
            model_path = config.MODEL_PATH
        
        self.model = joblib.load(model_path)
        self.is_trained = True
        print(f"Model loaded from {model_path}")
    
    def predict_weather_description(self, X):
        """
        Predict weather with human-readable description
        
        Args:
            X (array): Features to predict
            
        Returns:
            list: Weather condition descriptions
        """
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)
        
        results = []
        for pred, prob in zip(predictions, probabilities):
            weather = config.WEATHER_CONDITIONS[pred]
            confidence = prob[pred] * 100
            results.append({
                'weather': weather,
                'confidence': confidence,
                'probabilities': {config.WEATHER_CONDITIONS[i]: prob[i] * 100 for i in range(len(prob))}
            })
        
        return results
