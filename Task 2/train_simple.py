#!/usr/bin/env python3
"""
Simple MNIST training script that ensures model gets saved properly
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os

def train_mnist_model():
    print("Loading MNIST dataset...")
    
    # Load MNIST data
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist.data, mnist.target.astype(int)
    
    # Use a smaller subset for faster training
    # Take first 20000 samples for training, next 5000 for testing
    X_train = X[:20000] / 255.0  # Normalize
    y_train = y[:20000]
    X_test = X[20000:25000] / 255.0
    y_test = y[20000:25000]
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Create a simpler MLP model for faster training
    print("Creating and training model...")
    model = MLPClassifier(
        hidden_layer_sizes=(100, 50),  # Smaller network
        activation='relu',
        solver='adam',
        alpha=0.0001,
        max_iter=50,  # Fewer iterations for faster training
        random_state=42,
        verbose=True
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate the model
    print("Evaluating model...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Generate classification report
    report = classification_report(y_test, y_pred_test)
    print("\nClassification Report:")
    print(report)
    
    # Save the model
    print("Saving model...")
    with open('mnist_model_mlp.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Model saved as 'mnist_model_mlp.pkl'")
    
    # Save evaluation results
    results = {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'classification_report': report
    }
    
    with open('evaluation_results_mlp.pkl', 'wb') as f:
        pickle.dump(results, f)
    print("Results saved as 'evaluation_results_mlp.pkl'")
    
    return model, results

if __name__ == "__main__":
    try:
        model, results = train_mnist_model()
        print("\n" + "="*50)
        print("Training completed successfully!")
        print(f"Final Test Accuracy: {results['test_accuracy']:.4f}")
        print("You can now run the web app with: python app.py")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()