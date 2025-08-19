from flask import Flask, render_template, request, jsonify
import json
import os
from spam_classifier import EmailSpamClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import pandas as pd

app = Flask(__name__)

# Initialize the classifier
classifier = EmailSpamClassifier()

# Global variables to store model metrics
model_metrics = {}

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/train_model', methods=['POST'])
def train_model():
    """Train the spam classification model"""
    try:
        accuracy, report = classifier.train_model()
        
        global model_metrics
        model_metrics = {
            'accuracy': accuracy,
            'precision_spam': report['spam']['precision'],
            'recall_spam': report['spam']['recall'],
            'f1_spam': report['spam']['f1-score'],
            'precision_ham': report['ham']['precision'],
            'recall_ham': report['ham']['recall'],
            'f1_ham': report['ham']['f1-score']
        }
        
        return jsonify({
            'success': True,
            'message': 'Model trained successfully!',
            'accuracy': accuracy,
            'metrics': model_metrics
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error training model: {str(e)}'
        })

@app.route('/predict', methods=['POST'])
def predict_email():
    """Predict if an email is spam or not"""
    try:
        data = request.get_json()
        email_text = data.get('email_text', '')
        
        if not email_text.strip():
            return jsonify({
                'success': False,
                'message': 'Please enter an email text to classify.'
            })
        
        result = classifier.predict(email_text)
        
        return jsonify({
            'success': True,
            'prediction': result['prediction'],
            'confidence': round(result['confidence'], 4),
            'spam_probability': round(result['spam_probability'], 4),
            'ham_probability': round(result['ham_probability'], 4)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error making prediction: {str(e)}'
        })

@app.route('/get_metrics')
def get_metrics():
    """Get current model metrics"""
    return jsonify(model_metrics)

@app.route('/get_confusion_matrix')
def get_confusion_matrix():
    """Get confusion matrix image as base64"""
    try:
        if os.path.exists('confusion_matrix.png'):
            with open('confusion_matrix.png', 'rb') as f:
                img_data = base64.b64encode(f.read()).decode()
            return jsonify({
                'success': True,
                'image': f'data:image/png;base64,{img_data}'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Confusion matrix not found. Please train the model first.'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error loading confusion matrix: {str(e)}'
        })

@app.route('/sample_emails')
def get_sample_emails():
    """Get sample emails for testing"""
    samples = {
        'spam': [
            "URGENT! You have won $1000000! Click here to claim your prize now!",
            "FREE MONEY! No strings attached. Send your bank details immediately.",
            "ALERT: Your account will be suspended. Verify your details now.",
            "Amazing opportunity! Earn $5000 per week working from home!",
            "Get viagra cheap! No prescription needed!"
        ],
        'ham': [
            "Hi John, hope you're doing well. Let's catch up over coffee this weekend.",
            "Meeting scheduled for tomorrow at 2 PM in conference room A.",
            "Thank you for your purchase. Your order will be delivered in 3-5 days.",
            "Project update: We've completed phase 1 and moving to phase 2.",
            "The quarterly report is ready for review. Please find it attached."
        ]
    }
    return jsonify(samples)

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Try to load existing model
    if classifier.load_model():
        print("Existing model loaded successfully!")
    else:
        print("No existing model found. Please train a new model.")
    
    app.run(debug=True, host='0.0.0.0', port=5000)