"""
Quick Demo: California Housing Price Prediction
==============================================
A simplified version for quick testing and demonstration.
"""

from california_housing_predictor import CaliforniaHousingPredictor

def quick_demo():
    """Run a quick demonstration of the housing price predictor"""
    print("🏠 California Housing Price Prediction - Quick Demo")
    print("="*55)
    
    # Initialize and run the predictor
    predictor = CaliforniaHousingPredictor()
    
    # Load data
    print("📊 Loading data...")
    predictor.load_data()
    
    # Quick data cleaning
    print("🧹 Cleaning data...")
    predictor.clean_data()
    
    # Feature selection
    print("🎯 Selecting features...")
    selected_features = predictor.select_features(correlation_threshold=0.1)
    
    # Prepare data
    print("⚙️ Preparing data...")
    predictor.prepare_data(selected_features)
    
    # Train models (without hyperparameter tuning for speed)
    print("🤖 Training models...")
    model_scores, best_model = predictor.train_models()
    
    # Evaluate
    print("📈 Evaluating model...")
    metrics = predictor.evaluate_model()
    
    # Quick prediction examples
    print("\n🔮 Sample Predictions:")
    print("-" * 30)
    
    examples = [
        {
            'name': 'Luxury House',
            'features': {
                'MedInc': 8.0, 'HouseAge': 5.0, 'AveRooms': 8.0,
                'AveBedrms': 1.5, 'Population': 2000.0, 'AveOccup': 2.5,
                'Latitude': 37.8, 'Longitude': -122.4
            }
        },
        {
            'name': 'Average House',
            'features': {
                'MedInc': 4.0, 'HouseAge': 15.0, 'AveRooms': 5.5,
                'AveBedrms': 1.1, 'Population': 3500.0, 'AveOccup': 3.2,
                'Latitude': 34.0, 'Longitude': -118.0
            }
        },
        {
            'name': 'Budget House',
            'features': {
                'MedInc': 2.5, 'HouseAge': 25.0, 'AveRooms': 4.0,
                'AveBedrms': 1.0, 'Population': 5000.0, 'AveOccup': 4.0,
                'Latitude': 32.7, 'Longitude': -117.2
            }
        }
    ]
    
    for example in examples:
        price = predictor.predict_price(example['features'])
        print(f"{example['name']}: ${price*100000:,.2f}")
    
    print(f"\n✅ Model Performance Summary:")
    print(f"   R² Score: {metrics['test_r2']:.3f}")
    print(f"   RMSE: {metrics['test_rmse']:.3f}")
    print(f"   MAE: {metrics['test_mae']:.3f}")
    
    return predictor, metrics

if __name__ == "__main__":
    predictor, metrics = quick_demo()