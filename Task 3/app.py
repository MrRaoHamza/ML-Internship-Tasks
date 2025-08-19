"""
Flask Web Application for California Housing Price Prediction
===========================================================
Professional web interface to display ML model results and predictions.
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
from california_housing_predictor import CaliforniaHousingPredictor
import json

app = Flask(__name__)

# Global variables to store model and results
predictor = None
model_metrics = None
feature_importance = None
selected_features = None

def initialize_model():
    """Initialize and train the model"""
    global predictor, model_metrics, feature_importance, selected_features
    
    print("Initializing model...")
    predictor = CaliforniaHousingPredictor()
    
    # Load and process data
    predictor.load_data()
    predictor.clean_data()
    selected_features = predictor.select_features(correlation_threshold=0.1)
    predictor.prepare_data(selected_features)
    
    # Train models
    model_scores, best_model_name = predictor.train_models()
    model_metrics = predictor.evaluate_model()
    
    # Get feature importance if available
    if hasattr(predictor.model, 'feature_importances_'):
        feature_names = predictor.X_train.columns
        importances = predictor.model.feature_importances_
        feature_importance = dict(zip(feature_names, importances))
    
    print("Model initialized successfully!")
    return model_metrics, feature_importance

def create_plot_base64(plot_func):
    """Convert matplotlib plot to base64 string"""
    img = io.BytesIO()
    plot_func()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=150)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url

def create_correlation_heatmap():
    """Create correlation heatmap"""
    plt.figure(figsize=(10, 8))
    correlation_matrix = predictor.df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0, 
                fmt='.2f', square=True, linewidths=0.5)
    plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()

def create_feature_importance_plot():
    """Create feature importance plot"""
    if feature_importance:
        plt.figure(figsize=(10, 6))
        features = list(feature_importance.keys())
        importances = list(feature_importance.values())
        
        # Sort by importance
        sorted_idx = np.argsort(importances)[::-1]
        features = [features[i] for i in sorted_idx]
        importances = [importances[i] for i in sorted_idx]
        
        plt.barh(range(len(features)), importances, color='skyblue', edgecolor='navy')
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance Score', fontsize=12)
        plt.title('Feature Importance (Random Forest)', fontsize=16, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()

def create_price_distribution():
    """Create price distribution plot"""
    plt.figure(figsize=(10, 6))
    plt.hist(predictor.df['target'], bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.xlabel('House Price (in hundreds of thousands)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of House Prices in California', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

@app.route('/')
def index():
    """Main dashboard page"""
    global predictor, model_metrics, feature_importance
    
    if predictor is None:
        initialize_model()
    
    # Create plots
    correlation_plot = create_plot_base64(create_correlation_heatmap)
    importance_plot = create_plot_base64(create_feature_importance_plot) if feature_importance else None
    distribution_plot = create_plot_base64(create_price_distribution)
    
    # Prepare data for template
    dataset_info = {
        'total_samples': len(predictor.df),
        'features_count': len(selected_features),
        'target_mean': f"${predictor.df['target'].mean() * 100000:,.2f}",
        'target_std': f"${predictor.df['target'].std() * 100000:,.2f}"
    }
    
    return render_template('index.html', 
                         model_metrics=model_metrics,
                         feature_importance=feature_importance,
                         dataset_info=dataset_info,
                         correlation_plot=correlation_plot,
                         importance_plot=importance_plot,
                         distribution_plot=distribution_plot,
                         selected_features=selected_features)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get form data
        features = {
            'MedInc': float(request.form['MedInc']),
            'HouseAge': float(request.form['HouseAge']),
            'AveRooms': float(request.form['AveRooms']),
            'AveBedrms': float(request.form['AveBedrms']),
            'Population': float(request.form['Population']),
            'AveOccup': float(request.form['AveOccup']),
            'Latitude': float(request.form['Latitude']),
            'Longitude': float(request.form['Longitude'])
        }
        
        # Make prediction
        predicted_price = predictor.predict_price(features)
        predicted_price_dollars = predicted_price * 100000
        
        return jsonify({
            'success': True,
            'predicted_price': f"${predicted_price_dollars:,.2f}",
            'features_used': features
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/model-info')
def model_info():
    """API endpoint for model information"""
    return jsonify({
        'model_type': type(predictor.model).__name__,
        'features_count': len(selected_features),
        'training_samples': len(predictor.X_train),
        'test_samples': len(predictor.X_test),
        'metrics': model_metrics,
        'feature_importance': feature_importance
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)