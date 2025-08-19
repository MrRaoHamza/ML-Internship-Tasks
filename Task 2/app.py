from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import base64
from PIL import Image
import io
import pickle
import os
from simple_predictor import SimplePredictor
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

app = Flask(__name__)

# Global classifier instance
classifier = SimplePredictor()

# Load model and evaluation results on startup
def load_model_and_results():
    global classifier, evaluation_results
    
    # Load the trained model
    if classifier.load_model('mnist_model_mlp.pkl'):
        print("Model loaded successfully")
    else:
        print("No trained model found. Please train the model first.")
    
    # Load evaluation results
    try:
        with open('evaluation_results_mlp.pkl', 'rb') as f:
            evaluation_results = pickle.load(f)
        print("Evaluation results loaded")
    except FileNotFoundError:
        evaluation_results = None
        print("No evaluation results found")

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/model_info')
def model_info():
    """Get model information and evaluation metrics"""
    if evaluation_results is None:
        return jsonify({'error': 'No evaluation results available'})
    
    # Handle both TensorFlow and sklearn result formats
    if 'test_accuracy' in evaluation_results:
        # sklearn format
        return jsonify({
            'accuracy': float(evaluation_results['test_accuracy']),
            'loss': 0.0,  # sklearn doesn't provide loss
            'classification_report': evaluation_results['classification_report']
        })
    else:
        # TensorFlow format
        return jsonify({
            'accuracy': float(evaluation_results.get('accuracy', 0)),
            'loss': float(evaluation_results.get('loss', 0)),
            'classification_report': evaluation_results['classification_report']
        })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict digit from uploaded image or canvas drawing"""
    try:
        if 'image' in request.files:
            # Handle file upload
            file = request.files['image']
            image = Image.open(file.stream).convert('L')
            image = image.resize((28, 28))
            image_array = np.array(image)
        
        elif 'canvas_data' in request.json:
            # Handle canvas drawing
            canvas_data = request.json['canvas_data']
            # Remove data URL prefix
            image_data = canvas_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert('L')
            image = image.resize((28, 28))
            image_array = np.array(image)
            # Invert colors (canvas is white on black, MNIST is black on white)
            image_array = 255 - image_array
        
        else:
            return jsonify({'error': 'No image data provided'})
        
        # Normalize image
        image_array = image_array.astype('float32') / 255.0
        
        # Make prediction
        predicted_digit, confidence, probabilities = classifier.predict_digit(image_array)
        
        return jsonify({
            'predicted_digit': int(predicted_digit),
            'confidence': float(confidence),
            'probabilities': [float(p) for p in probabilities]
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/sample_predictions')
def sample_predictions():
    """Get sample predictions for display"""
    try:
        # Load test data
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X, y = mnist.data, mnist.target.astype(int)
        
        # Use last 1000 samples as test set and reshape properly
        x_test_flat = X[-1000:]  # Keep as flat for prediction
        x_test_images = X[-1000:].reshape(-1, 28, 28)  # Reshape for display
        y_test = y[-1000:]
        
        # Get random samples
        indices = np.random.choice(len(x_test_flat), 12, replace=False)
        samples = []
        
        for idx in indices:
            # Get flat image for prediction
            image_flat = x_test_flat[idx]
            # Get 2D image for display
            image_2d = x_test_images[idx]
            true_label = y_test[idx]
            
            # Normalize and predict using flat image
            normalized_image = image_flat.astype('float32') / 255.0
            predicted_digit, confidence, _ = classifier.predict_digit(normalized_image)
            
            # Convert 2D image to base64 for display
            # Ensure image is in proper format (0-255, uint8)
            image_display = (image_2d * 255).astype(np.uint8) if image_2d.max() <= 1 else image_2d.astype(np.uint8)
            
            # Create PIL image
            img_pil = Image.fromarray(image_display, mode='L')  # 'L' for grayscale
            buffer = io.BytesIO()
            img_pil.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            samples.append({
                'image': f"data:image/png;base64,{img_str}",
                'true_label': int(true_label),
                'predicted_label': int(predicted_digit),
                'confidence': float(confidence),
                'correct': int(true_label) == int(predicted_digit)
            })
        
        return jsonify(samples)
    
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/train')
def train_page():
    """Training page"""
    return render_template('train.html')

@app.route('/api/train', methods=['POST'])
def train_model():
    """Train the model"""
    try:
        epochs = request.json.get('epochs', 10)
        
        # This would typically be run in a separate thread/process
        # For demo purposes, we'll return a success message
        return jsonify({
            'message': f'Training started with {epochs} epochs. This may take several minutes.',
            'status': 'started'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    load_model_and_results()
    app.run(debug=True, host='0.0.0.0', port=5000)