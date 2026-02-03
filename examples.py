"""
Example usage of the Weather Prediction Model

This script demonstrates different ways to use the weather prediction model.
"""

from predict import predict_single_sample, predict_batch
import numpy as np


def example_1_simple_prediction():
    """Example 1: Simple single prediction"""
    print("=" * 60)
    print("Example 1: Simple Weather Prediction")
    print("=" * 60)
    
    result = predict_single_sample(
        temperature=28,      # 28°C
        humidity=55,         # 55%
        pressure=1015,       # 1015 hPa
        wind_speed=8,        # 8 km/h
        cloud_cover=30       # 30%
    )
    
    print(f"\nInput Parameters:")
    print(f"  Temperature: 28°C")
    print(f"  Humidity: 55%")
    print(f"  Pressure: 1015 hPa")
    print(f"  Wind Speed: 8 km/h")
    print(f"  Cloud Cover: 30%")
    
    print(f"\nPrediction Result:")
    print(f"  Weather: {result['weather']}")
    print(f"  Confidence: {result['confidence']:.2f}%")
    print()


def example_2_batch_prediction():
    """Example 2: Batch prediction for multiple samples"""
    print("=" * 60)
    print("Example 2: Batch Weather Predictions")
    print("=" * 60)
    
    # Multiple weather scenarios
    scenarios = np.array([
        [32, 35, 1020, 5, 10],    # Hot sunny day
        [20, 70, 1008, 12, 60],   # Mild cloudy day
        [16, 88, 1002, 18, 95],   # Cool rainy day
        [12, 92, 988, 28, 100]    # Cold stormy day
    ])
    
    results = predict_batch(scenarios)
    
    scenario_names = [
        "Hot Sunny Day",
        "Mild Cloudy Day", 
        "Cool Rainy Day",
        "Cold Stormy Day"
    ]
    
    for i, (name, result) in enumerate(zip(scenario_names, results)):
        print(f"\n{name}:")
        print(f"  Predicted: {result['weather']} ({result['confidence']:.1f}% confidence)")
    print()


def example_3_detailed_probabilities():
    """Example 3: Getting detailed probability distribution"""
    print("=" * 60)
    print("Example 3: Detailed Probability Analysis")
    print("=" * 60)
    
    result = predict_single_sample(
        temperature=25,
        humidity=60,
        pressure=1012,
        wind_speed=10,
        cloud_cover=45
    )
    
    print(f"\nConditions: T=25°C, H=60%, P=1012hPa, WS=10km/h, CC=45%")
    print(f"\nPredicted Weather: {result['weather']}")
    print(f"\nProbability Distribution:")
    
    # Sort probabilities for better display
    sorted_probs = sorted(
        result['probabilities'].items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    for condition, probability in sorted_probs:
        bar_length = int(probability / 2)
        bar = "█" * bar_length
        print(f"  {condition:10s} {probability:5.1f}% {bar}")
    print()


def main():
    """Run all examples"""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "WEATHER PREDICTION MODEL - EXAMPLES" + " " * 13 + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    
    try:
        example_1_simple_prediction()
        example_2_batch_prediction()
        example_3_detailed_probabilities()
        
        print("=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        print()
        
    except FileNotFoundError:
        print("\n❌ Error: Model files not found!")
        print("Please run 'python train_model.py' first to train the model.")
        print()


if __name__ == "__main__":
    main()
