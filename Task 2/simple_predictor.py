import pickle
import numpy as np
from PIL import Image

class SimplePredictor:
    def __init__(self):
        self.model = None
        
    def load_model(self, filepath='mnist_model_mlp.pkl'):
        """Load the trained model"""
        try:
            with open(filepath, 'rb') as f:
                self.model = pickle.load(f)
            print(f"Model loaded from {filepath}")
            return True
        except FileNotFoundError:
            print(f"Model file {filepath} not found")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict_digit(self, image):
        """Predict digit from image"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Ensure image is flattened and normalized
        if len(image.shape) == 2:
            image = image.flatten()
        
        # Normalize if needed
        if image.max() > 1:
            image = image.astype('float32') / 255.0
        
        # Reshape for prediction
        image = image.reshape(1, -1)
        
        # Get prediction
        prediction = self.model.predict(image)[0]
        
        # Get probabilities if available
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(image)[0]
            confidence = np.max(probabilities)
        else:
            probabilities = np.zeros(10)
            probabilities[prediction] = 1.0
            confidence = 1.0
        
        return int(prediction), float(confidence), probabilities.tolist()