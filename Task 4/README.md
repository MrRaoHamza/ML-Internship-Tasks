# Iris Flower Classification Dashboard

A complete machine learning solution for classifying Iris flowers with an interactive web dashboard.

## Features

- **Machine Learning Models**: Trains and compares Random Forest, Logistic Regression, and SVM models
- **Data Preprocessing**: Automatic feature scaling and train/test splitting
- **Model Evaluation**: Comprehensive accuracy metrics and confusion matrix visualization
- **Interactive Dashboard**: Web-based interface for making predictions and visualizing data
- **Real-time Predictions**: Input flower measurements and get instant species predictions
- **Data Visualization**: Interactive plots showing feature distributions and relationships

## Project Structure

```
Task 4/
├── iris_classifier.py      # Main ML training script
├── app.py                 # Flask web application
├── templates/
│   └── dashboard.html     # Web dashboard interface
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── best_iris_model.pkl   # Trained model (generated)
├── scaler.pkl           # Feature scaler (generated)
├── model_info.pkl       # Model metadata (generated)
└── confusion_matrix.png # Model evaluation plot (generated)
```

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Train the Model

First, run the machine learning training script:

```bash
python iris_classifier.py
```

This will:
- Load and preprocess the Iris dataset
- Train multiple ML models (Random Forest, Logistic Regression, SVM)
- Select the best performing model
- Save the trained model and preprocessing components
- Generate evaluation metrics and confusion matrix

### 2. Launch the Web Dashboard

Start the Flask web application:

```bash
python app.py
```

Then open your browser and navigate to: `http://localhost:5000`

## Dashboard Features

### 🔮 Prediction Panel
- Input iris flower measurements (sepal length/width, petal length/width)
- Get instant species predictions with confidence probabilities
- Visual probability bars for each species

### 📊 Dataset Information
- Total samples and feature count
- Class distribution statistics
- Model performance metrics

### 📈 Interactive Visualizations
- Scatter plot showing sepal vs petal length relationships
- Feature distribution histograms for each measurement
- Color-coded by species for easy pattern recognition

## Model Performance

The system trains three different models and automatically selects the best performer:

- **Random Forest**: Ensemble method, good for feature importance
- **Logistic Regression**: Linear classifier with probability outputs
- **SVM**: Support Vector Machine for complex decision boundaries

Typical accuracy: 95-100% on the Iris dataset

## API Endpoints

- `POST /predict` - Make species predictions
- `GET /api/dataset_info` - Get dataset statistics
- `GET /api/feature_distribution` - Get feature distribution plots
- `GET /api/scatter_plot` - Get scatter plot data

## Input Features

The model uses four measurements to classify iris flowers:

1. **Sepal Length** (cm) - Length of the outer petals
2. **Sepal Width** (cm) - Width of the outer petals  
3. **Petal Length** (cm) - Length of the inner petals
4. **Petal Width** (cm) - Width of the inner petals

## Output Classes

The model predicts one of three iris species:

- **Setosa** - Typically smaller flowers with distinct measurements
- **Versicolor** - Medium-sized flowers with intermediate features
- **Virginica** - Larger flowers with longer petals

## Technical Details

- **Framework**: Flask for web application
- **ML Library**: Scikit-learn for model training
- **Visualization**: Plotly for interactive charts
- **Frontend**: Bootstrap for responsive design
- **Data Processing**: Pandas and NumPy for data manipulation

## Example Usage

1. Run `python iris_classifier.py` to train models
2. Run `python app.py` to start the web server
3. Open browser to `http://localhost:5000`
4. Enter flower measurements in the prediction form
5. View results and explore data visualizations

The dashboard provides an intuitive interface for both technical and non-technical users to interact with the machine learning model and understand the iris classification problem.