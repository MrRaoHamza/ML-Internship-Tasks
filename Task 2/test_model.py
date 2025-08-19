#!/usr/bin/env python3
"""
Test script to verify the model is working correctly
"""

from simple_predictor import SimplePredictor
import numpy as np
from sklearn.datasets import fetch_openml
import pickle

def test_model():
    print("Testing MNIST model...")
    
    # Load the predictor
    predictor = SimplePredictor()
    
    # Load the model
    if not predictor.load_model('mnist_model_mlp.pkl'):
        print("Failed to load model!")
        return False
    
    # Load some test data
    print("Loading test data...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist.data, mnist.target.astype(int)
    
    # Test with a few samples
    test_indices = [0, 100, 200, 300, 400]
    
    print("\nTesting predictions:")
    print("-" * 40)
    
    for i, idx in enumerate(test_indices):
        image = X[idx]
        true_label = y[idx]
        
        # Reshape to 28x28 for display info
        image_2d = image.reshape(28, 28)
        
        try:
            # Make prediction
            pred_digit, confidence, probabilities = predictor.predict_digit(image)
            
            print(f"Sample {i+1}:")
            print(f"  True label: {true_label}")
            print(f"  Predicted: {pred_digit}")
            print(f"  Confidence: {confidence:.4f}")
            print(f"  Correct: {'✓' if pred_digit == true_label else '✗'}")
            print()
            
        except Exception as e:
            print(f"Error predicting sample {i+1}: {e}")
            return False
    
    # Test evaluation results
    try:
        with open('evaluation_results_mlp.pkl', 'rb') as f:
            results = pickle.load(f)
        
        print("Model Performance:")
        print(f"  Test Accuracy: {results['test_accuracy']:.4f}")
        print(f"  Train Accuracy: {results['train_accuracy']:.4f}")
        
    except Exception as e:
        print(f"Could not load evaluation results: {e}")
    
    print("\nModel test completed successfully!")
    return True

if __name__ == "__main__":
    test_model()