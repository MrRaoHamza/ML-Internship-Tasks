"""
Simplified Flask App for California Housing Price Prediction
===========================================================
A simpler version with better error handling and debugging.
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
import traceback

app = Flask(__name__)

# Global variables
predictor = None
model_ready = False
error_message = None

def create_plot_base64(plot_func):
    """Convert matplotlib plot to base64 string"""
    try:
        img = io.BytesIO()
        plt.figure(figsize=(10, 6))
        plot_func()
        plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        return plot_url
    except Exception as e:
        print(f"Error creating plot: {e}")
        return None

def initialize_model():
    """Initialize the model with error handling"""
    global predictor, model_ready, error_message
    
    try:
        print("Starting model initialization...")
        
        from california_housing_predictor import CaliforniaHousingPredictor
        predictor = CaliforniaHousingPredictor()
        
        print("Loading data...")
        predictor.load_data()
        
        print("Cleaning data...")
        predictor.clean_data()
        
        print("Selecting features...")
        selected_features = predictor.select_features(correlation_threshold=0.1)
        
        print("Preparing data...")
        predictor.prepare_data(selected_features)
        
        print("Training models...")
        model_scores, best_model_name = predictor.train_models()
        
        print("Evaluating model...")
        model_metrics = predictor.evaluate_model()
        
        model_ready = True
        print("Model initialization complete!")
        
        return True, model_metrics, selected_features
        
    except Exception as e:
        error_message = f"Error initializing model: {str(e)}\n{traceback.format_exc()}"
        print(error_message)
        model_ready = False
        return False, None, None

@app.route('/')
def index():
    """Main page"""
    global predictor, model_ready, error_message
    
    if not model_ready:
        success, model_metrics, selected_features = initialize_model()
        if not success:
            return f"""
            <html>
            <head><title>Error</title></head>
            <body>
                <h1>Model Initialization Error</h1>
                <pre>{error_message}</pre>
                <p><a href="/">Retry</a></p>
            </body>
            </html>
            """
    else:
        # Get metrics from the already trained model
        train_pred = predictor.model.predict(predictor.X_train_scaled)
        test_pred = predictor.model.predict(predictor.X_test_scaled)
        
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        
        model_metrics = {
            'test_r2': r2_score(predictor.y_test, test_pred),
            'test_rmse': np.sqrt(mean_squared_error(predictor.y_test, test_pred)),
            'test_mae': mean_absolute_error(predictor.y_test, test_pred)
        }
        
        selected_features = list(predictor.X_train.columns)
    
    # Create simple plots
    def create_correlation_plot():
        correlation_matrix = predictor.df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0, fmt='.2f')
        plt.title('Feature Correlation Matrix')
    
    def create_distribution_plot():
        plt.hist(predictor.df['target'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('House Price (hundreds of thousands)')
        plt.ylabel('Frequency')
        plt.title('Distribution of House Prices')
    
    # Generate plots
    correlation_plot = create_plot_base64(create_correlation_plot)
    distribution_plot = create_plot_base64(create_distribution_plot)
    
    # Feature importance
    feature_importance = None
    if hasattr(predictor.model, 'feature_importances_'):
        feature_names = predictor.X_train.columns
        importances = predictor.model.feature_importances_
        feature_importance = dict(zip(feature_names, importances))
    
    # Dataset info
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
                         distribution_plot=distribution_plot,
                         selected_features=selected_features)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        if not model_ready:
            return jsonify({'success': False, 'error': 'Model not ready'})
        
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

@app.route('/status')
def status():
    """Status endpoint for debugging"""
    return jsonify({
        'model_ready': model_ready,
        'error_message': error_message,
        'predictor_exists': predictor is not None
    })

if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(debug=True, host='127.0.0.1', port=5000)