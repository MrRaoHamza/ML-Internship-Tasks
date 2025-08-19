# California Housing Price Prediction

A professional machine learning web application that predicts house prices in California using various features like location, number of rooms, and population data.

## 🌟 Features

### Machine Learning Pipeline
- **Data Loading**: Uses the built-in California housing dataset from scikit-learn
- **Data Cleaning**: Handles outliers and creates engineered features
- **Feature Selection**: Selects important features based on correlation with target
- **Model Training**: Compares Linear Regression and Random Forest models
- **Model Evaluation**: Comprehensive evaluation with multiple metrics and visualizations

### Professional Web Interface
- **Interactive Dashboard**: Beautiful, responsive web interface
- **Real-time Predictions**: Instant house price predictions with custom inputs
- **Data Visualizations**: Interactive charts showing correlations, distributions, and feature importance
- **Model Metrics**: Live display of R², RMSE, MAE, and other performance indicators
- **Sample Data**: Quick-load buttons for luxury, average, and budget house examples

## 🚀 Quick Start

### Option 1: Web Application (Recommended)
1. **Double-click to run:**
   ```
   start_web_app.bat  (Windows)
   ```
   
2. **Or run manually:**
   ```bash
   python run_web_app.py
   ```

3. **The web app will automatically:**
   - Install missing dependencies
   - Train the ML model
   - Open your browser to http://localhost:5000

### Option 2: Command Line
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the quick demo:**
   ```bash
   python quick_demo.py
   ```

3. **Run the full analysis:**
   ```bash
   python california_housing_predictor.py
   ```

## Dataset Features

The California housing dataset includes:
- **MedInc**: Median income in block group
- **HouseAge**: Median house age in block group
- **AveRooms**: Average number of rooms per household
- **AveBedrms**: Average number of bedrooms per household
- **Population**: Block group population
- **AveOccup**: Average number of household members
- **Latitude**: Block group latitude
- **Longitude**: Block group longitude

## Engineered Features

The model creates additional features:
- **rooms_per_household**: AveRooms / AveOccup
- **bedrooms_per_room**: AveBedrms / AveRooms
- **population_per_household**: Population / HouseAge

## Model Performance

The Random Forest model typically achieves:
- **R² Score**: ~0.80-0.85
- **RMSE**: ~0.50-0.60
- **MAE**: ~0.35-0.45

## Usage Example

```python
from california_housing_predictor import CaliforniaHousingPredictor

# Initialize predictor
predictor = CaliforniaHousingPredictor()

# Train the model (full pipeline)
predictor.load_data()
predictor.clean_data()
features = predictor.select_features()
predictor.prepare_data(features)
predictor.train_models()

# Make a prediction
house_features = {
    'MedInc': 5.0,
    'HouseAge': 10.0,
    'AveRooms': 6.0,
    'AveBedrms': 1.2,
    'Population': 3000.0,
    'AveOccup': 3.0,
    'Latitude': 34.0,
    'Longitude': -118.0
}

predicted_price = predictor.predict_price(house_features)
print(f"Predicted price: ${predicted_price*100000:.2f}")
```

## 📁 Project Structure

```
├── app.py                          # Flask web application
├── california_housing_predictor.py # Main ML pipeline class
├── run_web_app.py                  # Web app launcher
├── start_web_app.bat              # Windows batch launcher
├── quick_demo.py                   # Command-line demo
├── requirements.txt                # Python dependencies
├── templates/
│   └── index.html                 # Professional web interface
├── static/                        # Static assets (auto-generated)
└── README.md                      # This file
```

## 🎯 Web Interface Features

### Dashboard Components
- **Performance Metrics**: Live R², RMSE, MAE scores
- **Interactive Prediction Form**: Input custom house features
- **Data Visualizations**: 
  - Price distribution histogram
  - Feature correlation heatmap  
  - Feature importance chart
- **Sample Data Buttons**: Quick-load luxury, average, budget examples
- **Responsive Design**: Works on desktop, tablet, and mobile

### Model Performance
The Random Forest model typically achieves:
- **R² Score**: ~0.80-0.85 (80-85% accuracy)
- **RMSE**: ~0.50-0.60
- **MAE**: ~0.35-0.45

## 🔧 Technical Details

### Visualizations Generated
- Data exploration plots
- Model evaluation plots  
- Feature correlation heatmap
- Actual vs predicted scatter plot
- Residuals analysis
- Feature importance rankings