"""
California Housing Price Prediction Model
========================================
This script builds a machine learning model to predict house prices using the California housing dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class CaliforniaHousingPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self):
        """Load the California housing dataset"""
        print("Loading California housing dataset...")
        housing = fetch_california_housing()
        
        # Create DataFrame
        self.df = pd.DataFrame(housing.data, columns=housing.feature_names)
        self.df['target'] = housing.target
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"Features: {list(housing.feature_names)}")
        return self.df
    
    def explore_data(self):
        """Explore and visualize the dataset"""
        print("\n" + "="*50)
        print("DATA EXPLORATION")
        print("="*50)
        
        # Basic info
        print("\nDataset Info:")
        print(self.df.info())
        
        print("\nBasic Statistics:")
        print(self.df.describe())
        
        # Check for missing values
        print(f"\nMissing values:\n{self.df.isnull().sum()}")
        
        # Create visualizations
        plt.figure(figsize=(15, 10))
        
        # Distribution of target variable
        plt.subplot(2, 3, 1)
        plt.hist(self.df['target'], bins=50, alpha=0.7)
        plt.title('Distribution of House Prices')
        plt.xlabel('Price (in hundreds of thousands)')
        plt.ylabel('Frequency')
        
        # Correlation heatmap
        plt.subplot(2, 3, 2)
        correlation_matrix = self.df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Feature Correlation Matrix')
        
        # Feature vs target scatter plots
        important_features = ['MedInc', 'AveRooms', 'Population', 'Latitude', 'Longitude']
        for i, feature in enumerate(important_features[:3]):
            plt.subplot(2, 3, i+3)
            plt.scatter(self.df[feature], self.df['target'], alpha=0.5)
            plt.xlabel(feature)
            plt.ylabel('House Price')
            plt.title(f'{feature} vs House Price')
        
        plt.tight_layout()
        plt.savefig('data_exploration.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return correlation_matrix
    
    def clean_data(self):
        """Clean and preprocess the data"""
        print("\n" + "="*50)
        print("DATA CLEANING")
        print("="*50)
        
        # Check for outliers using IQR method
        print("Checking for outliers...")
        Q1 = self.df.quantile(0.25)
        Q3 = self.df.quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outlier bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Count outliers for each feature
        outliers_count = {}
        for column in self.df.columns:
            outliers = ((self.df[column] < lower_bound[column]) | 
                       (self.df[column] > upper_bound[column])).sum()
            outliers_count[column] = outliers
            
        print("Outliers per feature:")
        for feature, count in outliers_count.items():
            print(f"{feature}: {count} outliers")
        
        # Remove extreme outliers (optional - keeping for now to preserve data)
        initial_shape = self.df.shape[0]
        
        # Create engineered features
        print("\nCreating engineered features...")
        self.df['rooms_per_household'] = self.df['AveRooms'] / self.df['AveOccup']
        self.df['bedrooms_per_room'] = self.df['AveBedrms'] / self.df['AveRooms']
        self.df['population_per_household'] = self.df['Population'] / self.df['HouseAge']
        
        print(f"Added 3 engineered features")
        print(f"Final dataset shape: {self.df.shape}")
        
        return self.df
    
    def select_features(self, correlation_threshold=0.1):
        """Select important features based on correlation with target"""
        print("\n" + "="*50)
        print("FEATURE SELECTION")
        print("="*50)
        
        # Calculate correlation with target
        correlations = self.df.corr()['target'].abs().sort_values(ascending=False)
        
        print("Feature correlations with target:")
        for feature, corr in correlations.items():
            if feature != 'target':
                print(f"{feature}: {corr:.3f}")
        
        # Select features above threshold
        selected_features = correlations[correlations > correlation_threshold].index.tolist()
        selected_features.remove('target')  # Remove target from features
        
        print(f"\nSelected {len(selected_features)} features above correlation threshold {correlation_threshold}:")
        print(selected_features)
        
        return selected_features
    
    def prepare_data(self, selected_features):
        """Prepare data for training"""
        print("\n" + "="*50)
        print("DATA PREPARATION")
        print("="*50)
        
        # Separate features and target
        X = self.df[selected_features]
        y = self.df['target']
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set size: {self.X_train.shape[0]}")
        print(f"Test set size: {self.X_test.shape[0]}")
        print(f"Number of features: {self.X_train.shape[1]}")
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def train_models(self):
        """Train multiple models and compare performance"""
        print("\n" + "="*50)
        print("MODEL TRAINING")
        print("="*50)
        
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        model_scores = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(self.X_train_scaled, self.y_train)
            
            # Cross-validation
            cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, 
                                      cv=5, scoring='r2')
            
            # Predictions
            train_pred = model.predict(self.X_train_scaled)
            test_pred = model.predict(self.X_test_scaled)
            
            # Calculate metrics
            train_r2 = r2_score(self.y_train, train_pred)
            test_r2 = r2_score(self.y_test, test_pred)
            train_rmse = np.sqrt(mean_squared_error(self.y_train, train_pred))
            test_rmse = np.sqrt(mean_squared_error(self.y_test, test_pred))
            test_mae = mean_absolute_error(self.y_test, test_pred)
            
            model_scores[name] = {
                'model': model,
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std(),
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'test_predictions': test_pred
            }
            
            print(f"Cross-validation R²: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            print(f"Train R²: {train_r2:.3f}")
            print(f"Test R²: {test_r2:.3f}")
            print(f"Test RMSE: {test_rmse:.3f}")
            print(f"Test MAE: {test_mae:.3f}")
        
        # Select best model
        best_model_name = max(model_scores.keys(), 
                             key=lambda x: model_scores[x]['test_r2'])
        self.model = model_scores[best_model_name]['model']
        
        print(f"\nBest model: {best_model_name}")
        
        return model_scores, best_model_name
    
    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning for Random Forest"""
        print("\n" + "="*50)
        print("HYPERPARAMETER TUNING")
        print("="*50)
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Grid search
        rf = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='r2', 
                                 n_jobs=-1, verbose=1)
        
        print("Performing grid search...")
        grid_search.fit(self.X_train_scaled, self.y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
        
        # Train final model with best parameters
        self.model = grid_search.best_estimator_
        
        return grid_search.best_estimator_
    
    def evaluate_model(self):
        """Evaluate the final model"""
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        # Make predictions
        train_pred = self.model.predict(self.X_train_scaled)
        test_pred = self.model.predict(self.X_test_scaled)
        
        # Calculate metrics
        train_r2 = r2_score(self.y_train, train_pred)
        test_r2 = r2_score(self.y_test, test_pred)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, test_pred))
        test_mae = mean_absolute_error(self.y_test, test_pred)
        
        print(f"Final Model Performance:")
        print(f"Train R²: {train_r2:.3f}")
        print(f"Test R²: {test_r2:.3f}")
        print(f"Test RMSE: {test_rmse:.3f}")
        print(f"Test MAE: {test_mae:.3f}")
        
        # Feature importance (if Random Forest)
        if hasattr(self.model, 'feature_importances_'):
            feature_names = self.X_train.columns
            importances = self.model.feature_importances_
            
            print(f"\nFeature Importances:")
            for name, importance in zip(feature_names, importances):
                print(f"{name}: {importance:.3f}")
        
        # Create evaluation plots
        self.plot_results(test_pred)
        
        return {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae
        }
    
    def plot_results(self, predictions):
        """Create visualization plots for model evaluation"""
        plt.figure(figsize=(15, 5))
        
        # Actual vs Predicted
        plt.subplot(1, 3, 1)
        plt.scatter(self.y_test, predictions, alpha=0.5)
        plt.plot([self.y_test.min(), self.y_test.max()], 
                [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Prices')
        plt.ylabel('Predicted Prices')
        plt.title('Actual vs Predicted Prices')
        
        # Residuals plot
        plt.subplot(1, 3, 2)
        residuals = self.y_test - predictions
        plt.scatter(predictions, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Prices')
        plt.ylabel('Residuals')
        plt.title('Residuals Plot')
        
        # Distribution of residuals
        plt.subplot(1, 3, 3)
        plt.hist(residuals, bins=30, alpha=0.7)
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Distribution of Residuals')
        
        plt.tight_layout()
        plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict_price(self, features_dict):
        """Make a prediction for new data"""
        # Convert to DataFrame
        new_data = pd.DataFrame([features_dict])
        
        # Add engineered features if they exist
        if 'AveRooms' in features_dict and 'AveOccup' in features_dict:
            new_data['rooms_per_household'] = new_data['AveRooms'] / new_data['AveOccup']
        if 'AveBedrms' in features_dict and 'AveRooms' in features_dict:
            new_data['bedrooms_per_room'] = new_data['AveBedrms'] / new_data['AveRooms']
        if 'Population' in features_dict and 'HouseAge' in features_dict:
            new_data['population_per_household'] = new_data['Population'] / new_data['HouseAge']
        
        # Select only the features used in training
        feature_columns = self.X_train.columns
        new_data_selected = new_data[feature_columns]
        
        # Scale the features
        new_data_scaled = self.scaler.transform(new_data_selected)
        
        # Make prediction
        prediction = self.model.predict(new_data_scaled)[0]
        
        return prediction

def main():
    """Main execution function"""
    print("California Housing Price Prediction")
    print("="*50)
    
    # Initialize predictor
    predictor = CaliforniaHousingPredictor()
    
    # Load and explore data
    df = predictor.load_data()
    correlation_matrix = predictor.explore_data()
    
    # Clean data
    cleaned_df = predictor.clean_data()
    
    # Select features
    selected_features = predictor.select_features(correlation_threshold=0.1)
    
    # Prepare data
    X_train_scaled, X_test_scaled, y_train, y_test = predictor.prepare_data(selected_features)
    
    # Train models
    model_scores, best_model_name = predictor.train_models()
    
    # Hyperparameter tuning (optional - comment out for faster execution)
    # predictor.hyperparameter_tuning()
    
    # Evaluate final model
    final_metrics = predictor.evaluate_model()
    
    # Example prediction
    print("\n" + "="*50)
    print("EXAMPLE PREDICTION")
    print("="*50)
    
    example_house = {
        'MedInc': 5.0,  # Median income
        'HouseAge': 10.0,  # House age
        'AveRooms': 6.0,  # Average rooms
        'AveBedrms': 1.2,  # Average bedrooms
        'Population': 3000.0,  # Population
        'AveOccup': 3.0,  # Average occupancy
        'Latitude': 34.0,  # Latitude
        'Longitude': -118.0  # Longitude
    }
    
    predicted_price = predictor.predict_price(example_house)
    print(f"Predicted price for example house: ${predicted_price*100000:.2f}")
    
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Best model: {best_model_name}")
    print(f"Test R² Score: {final_metrics['test_r2']:.3f}")
    print(f"Test RMSE: {final_metrics['test_rmse']:.3f}")
    print(f"Test MAE: {final_metrics['test_mae']:.3f}")
    print("\nModel successfully trained and evaluated!")

if __name__ == "__main__":
    main()