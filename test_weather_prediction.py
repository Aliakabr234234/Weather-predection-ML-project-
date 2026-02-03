"""
Unit Tests for Weather Prediction ML Project

Run with: python -m pytest test_weather_prediction.py -v
or: python test_weather_prediction.py
"""

import unittest
import numpy as np
import pandas as pd
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import (
    generate_sample_data,
    clean_data,
    prepare_features,
    split_data,
    scale_features
)
from weather_predictor import WeatherPredictor
import config


class TestDataPreprocessing(unittest.TestCase):
    """Test cases for data preprocessing module"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sample_data = generate_sample_data(n_samples=100)
    
    def test_generate_sample_data(self):
        """Test sample data generation"""
        data = generate_sample_data(n_samples=100)
        self.assertEqual(len(data), 100)
        self.assertIn('temperature', data.columns)
        self.assertIn('humidity', data.columns)
        self.assertIn('weather_condition', data.columns)
    
    def test_clean_data(self):
        """Test data cleaning"""
        cleaned = clean_data(self.sample_data)
        self.assertIsNotNone(cleaned)
        self.assertGreater(len(cleaned), 0)
        # Check no missing values in target
        self.assertEqual(cleaned[config.TARGET_COLUMN].isnull().sum(), 0)
    
    def test_prepare_features(self):
        """Test feature preparation"""
        X, y = prepare_features(self.sample_data)
        self.assertEqual(X.shape[0], y.shape[0])
        self.assertEqual(X.shape[1], len(config.FEATURE_COLUMNS))
    
    def test_split_data(self):
        """Test data splitting"""
        X, y = prepare_features(self.sample_data)
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
        
        total_samples = len(X)
        self.assertAlmostEqual(len(X_train) / total_samples, 0.8, delta=0.1)
        self.assertAlmostEqual(len(X_test) / total_samples, 0.2, delta=0.1)
    
    def test_scale_features(self):
        """Test feature scaling"""
        X, y = prepare_features(self.sample_data)
        X_train, X_test, _, _ = split_data(X, y)
        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
        
        self.assertEqual(X_train_scaled.shape, X_train.shape)
        self.assertEqual(X_test_scaled.shape, X_test.shape)
        
        # Check scaling worked (mean ~0, std ~1)
        self.assertAlmostEqual(np.mean(X_train_scaled), 0, delta=0.5)
        self.assertAlmostEqual(np.std(X_train_scaled), 1, delta=0.5)


class TestWeatherPredictor(unittest.TestCase):
    """Test cases for weather predictor model"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.data = generate_sample_data(n_samples=200)
        self.cleaned = clean_data(self.data)
        self.X, self.y = prepare_features(self.cleaned)
        self.X_train, self.X_test, self.y_train, self.y_test = split_data(
            self.X, self.y, test_size=0.2
        )
        self.X_train_scaled, self.X_test_scaled, self.scaler = scale_features(
            self.X_train, self.X_test
        )
        self.predictor = WeatherPredictor(n_estimators=10)  # Small for speed
    
    def test_model_initialization(self):
        """Test model initialization"""
        self.assertIsNotNone(self.predictor.model)
        self.assertFalse(self.predictor.is_trained)
    
    def test_model_training(self):
        """Test model training"""
        self.predictor.train(self.X_train_scaled, self.y_train)
        self.assertTrue(self.predictor.is_trained)
    
    def test_model_prediction(self):
        """Test model prediction"""
        self.predictor.train(self.X_train_scaled, self.y_train)
        predictions = self.predictor.predict(self.X_test_scaled)
        
        self.assertEqual(len(predictions), len(self.X_test_scaled))
        self.assertTrue(all(p in range(4) for p in predictions))
    
    def test_prediction_probabilities(self):
        """Test prediction probabilities"""
        self.predictor.train(self.X_train_scaled, self.y_train)
        probas = self.predictor.predict_proba(self.X_test_scaled)
        
        self.assertEqual(probas.shape[0], len(self.X_test_scaled))
        self.assertEqual(probas.shape[1], 4)  # 4 classes
        
        # Check probabilities sum to 1
        for proba in probas:
            self.assertAlmostEqual(np.sum(proba), 1.0, places=5)
    
    def test_model_evaluation(self):
        """Test model evaluation"""
        self.predictor.train(self.X_train_scaled, self.y_train)
        metrics = self.predictor.evaluate(self.X_test_scaled, self.y_test)
        
        self.assertIn('accuracy', metrics)
        self.assertIn('confusion_matrix', metrics)
        self.assertGreater(metrics['accuracy'], 0)
        self.assertLessEqual(metrics['accuracy'], 1)
    
    def test_feature_importance(self):
        """Test feature importance"""
        self.predictor.train(self.X_train_scaled, self.y_train)
        importance = self.predictor.get_feature_importance()
        
        self.assertEqual(len(importance), len(config.FEATURE_COLUMNS))
        
        # Check all importances sum to ~1
        total_importance = sum(importance.values())
        self.assertAlmostEqual(total_importance, 1.0, places=5)
    
    def test_weather_description_prediction(self):
        """Test weather description prediction"""
        self.predictor.train(self.X_train_scaled, self.y_train)
        results = self.predictor.predict_weather_description(self.X_test_scaled[:5])
        
        self.assertEqual(len(results), 5)
        for result in results:
            self.assertIn('weather', result)
            self.assertIn('confidence', result)
            self.assertIn('probabilities', result)
            self.assertIn(result['weather'], config.WEATHER_CONDITIONS.values())


class TestConfiguration(unittest.TestCase):
    """Test cases for configuration"""
    
    def test_config_constants(self):
        """Test configuration constants"""
        self.assertEqual(len(config.FEATURE_COLUMNS), 5)
        self.assertEqual(len(config.WEATHER_CONDITIONS), 4)
        self.assertIsInstance(config.RANDOM_STATE, int)
        self.assertIsInstance(config.TEST_SIZE, float)


class TestEndToEnd(unittest.TestCase):
    """End-to-end integration tests"""
    
    def test_complete_workflow(self):
        """Test complete training and prediction workflow"""
        # Generate data
        data = generate_sample_data(n_samples=200)
        
        # Preprocess
        cleaned = clean_data(data)
        X, y = prepare_features(cleaned)
        X_train, X_test, y_train, y_test = split_data(X, y)
        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
        
        # Train
        predictor = WeatherPredictor(n_estimators=10)
        predictor.train(X_train_scaled, y_train)
        
        # Predict
        predictions = predictor.predict(X_test_scaled)
        
        # Verify
        self.assertEqual(len(predictions), len(X_test_scaled))
        self.assertTrue(all(p in range(4) for p in predictions))
        
        # Evaluate
        metrics = predictor.evaluate(X_test_scaled, y_test)
        self.assertGreater(metrics['accuracy'], 0.3)  # Should be better than random


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestDataPreprocessing))
    suite.addTests(loader.loadTestsFromTestCase(TestWeatherPredictor))
    suite.addTests(loader.loadTestsFromTestCase(TestConfiguration))
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEnd))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    exit_code = run_tests()
    sys.exit(exit_code)
