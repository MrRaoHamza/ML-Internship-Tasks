from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import plotly.graph_objs as go
import plotly.utils
import json

app = Flask(__name__)

# Load the trained model and scaler
try:
    model = joblib.load('best_iris_model.pkl')
    scaler = joblib.load('scaler.pkl')
    model_info = joblib.load('model_info.pkl')
    print("Model loaded successfully!")
except:
    print("Model files not found. Please run iris_classifier.py first.")
    model = None
    scaler = None
    model_info = None

# Load iris dataset for visualization
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target
iris_df['species_name'] = [iris.target_names[i] for i in iris.target]

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for making predictions"""
    try:
        data = request.get_json()
        
        # Extract features
        sepal_length = float(data['sepal_length'])
        sepal_width = float(data['sepal_width'])
        petal_length = float(data['petal_length'])
        petal_width = float(data['petal_width'])
        
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        # Make prediction
        if model_info['model_name'] in ['Logistic Regression', 'SVM']:
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]
            probabilities = model.predict_proba(features_scaled)[0]
        else:
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
        
        # Prepare response
        result = {
            'prediction': iris.target_names[prediction],
            'probabilities': {
                iris.target_names[i]: float(prob) for i, prob in enumerate(probabilities)
            },
            'input_features': {
                'sepal_length': sepal_length,
                'sepal_width': sepal_width,
                'petal_length': petal_length,
                'petal_width': petal_width
            }
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/dataset_info')
def dataset_info():
    """API endpoint for dataset information"""
    info = {
        'total_samples': len(iris_df),
        'features': list(iris.feature_names),
        'classes': list(iris.target_names),
        'class_distribution': iris_df['species_name'].value_counts().to_dict()
    }
    return jsonify(info)

@app.route('/api/feature_distribution')
def feature_distribution():
    """API endpoint for feature distribution plots"""
    plots = {}
    
    for feature in iris.feature_names:
        fig = go.Figure()
        
        for species in iris.target_names:
            species_data = iris_df[iris_df['species_name'] == species][feature]
            fig.add_trace(go.Histogram(
                x=species_data,
                name=species,
                opacity=0.7,
                nbinsx=15
            ))
        
        fig.update_layout(
            title=f'Distribution of {feature}',
            xaxis_title=feature,
            yaxis_title='Frequency',
            barmode='overlay',
            height=400
        )
        
        plots[feature] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return jsonify(plots)

@app.route('/api/scatter_plot')
def scatter_plot():
    """API endpoint for scatter plot"""
    fig = go.Figure()
    
    colors = ['red', 'blue', 'green']
    
    for i, species in enumerate(iris.target_names):
        species_data = iris_df[iris_df['species_name'] == species]
        fig.add_trace(go.Scatter(
            x=species_data['sepal length (cm)'],
            y=species_data['petal length (cm)'],
            mode='markers',
            name=species,
            marker=dict(color=colors[i], size=8)
        ))
    
    fig.update_layout(
        title='Sepal Length vs Petal Length',
        xaxis_title='Sepal Length (cm)',
        yaxis_title='Petal Length (cm)',
        height=500
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

if __name__ == '__main__':
    app.run(debug=True, port=5000)