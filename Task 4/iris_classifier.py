import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class IrisClassifier:
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42),
            'SVM': SVC(random_state=42, probability=True)
        }
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_and_preprocess_data(self):
        """Load and preprocess the Iris dataset"""
        # Load the dataset
        iris = load_iris()
        self.X = pd.DataFrame(iris.data, columns=iris.feature_names)
        self.y = pd.DataFrame(iris.target, columns=['species'])
        self.target_names = iris.target_names
        
        print("Dataset loaded successfully!")
        print(f"Dataset shape: {self.X.shape}")
        print(f"Features: {list(self.X.columns)}")
        print(f"Target classes: {self.target_names}")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y.values.ravel(), test_size=0.2, random_state=42, stratify=self.y
        )
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        return self.X, self.y
    
    def train_models(self):
        """Train multiple models and select the best one"""
        results = {}
        
        for name, model in self.models.items():
            if name == 'Logistic Regression' or name == 'SVM':
                # Use scaled data for models that benefit from scaling
                model.fit(self.X_train_scaled, self.y_train)
                y_pred = model.predict(self.X_test_scaled)
            else:
                # Random Forest doesn't require scaling
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
            
            accuracy = accuracy_score(self.y_test, y_pred)
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'predictions': y_pred
            }
            
            print(f"{name} Accuracy: {accuracy:.4f}")
        
        # Select best model
        self.best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        self.best_model = results[self.best_model_name]['model']
        
        print(f"\nBest Model: {self.best_model_name} with accuracy: {results[self.best_model_name]['accuracy']:.4f}")
        
        return results
    
    def evaluate_model(self):
        """Evaluate the best model"""
        if self.best_model_name in ['Logistic Regression', 'SVM']:
            y_pred = self.best_model.predict(self.X_test_scaled)
        else:
            y_pred = self.best_model.predict(self.X_test)
        
        accuracy = accuracy_score(self.y_test, y_pred)
        
        print(f"\n=== Model Evaluation: {self.best_model_name} ===")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred, target_names=self.target_names))
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.target_names, yticklabels=self.target_names)
        plt.title(f'Confusion Matrix - {self.best_model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return accuracy, y_pred
    
    def save_model(self):
        """Save the trained model and scaler"""
        joblib.dump(self.best_model, 'best_iris_model.pkl')
        joblib.dump(self.scaler, 'scaler.pkl')
        
        # Save model info
        model_info = {
            'model_name': self.best_model_name,
            'feature_names': list(self.X.columns),
            'target_names': list(self.target_names)
        }
        joblib.dump(model_info, 'model_info.pkl')
        
        print(f"Model saved successfully!")
    
    def predict(self, features):
        """Make predictions on new data"""
        features_array = np.array(features).reshape(1, -1)
        
        if self.best_model_name in ['Logistic Regression', 'SVM']:
            features_scaled = self.scaler.transform(features_array)
            prediction = self.best_model.predict(features_scaled)[0]
            probabilities = self.best_model.predict_proba(features_scaled)[0]
        else:
            prediction = self.best_model.predict(features_array)[0]
            probabilities = self.best_model.predict_proba(features_array)[0]
        
        return prediction, probabilities

def main():
    # Initialize classifier
    classifier = IrisClassifier()
    
    # Load and preprocess data
    X, y = classifier.load_and_preprocess_data()
    
    # Train models
    results = classifier.train_models()
    
    # Evaluate best model
    accuracy, predictions = classifier.evaluate_model()
    
    # Save model
    classifier.save_model()
    
    # Example prediction
    print("\n=== Example Prediction ===")
    sample_features = [5.1, 3.5, 1.4, 0.2]  # Example iris features
    prediction, probabilities = classifier.predict(sample_features)
    
    print(f"Input features: {sample_features}")
    print(f"Predicted class: {classifier.target_names[prediction]}")
    print("Probabilities:")
    for i, prob in enumerate(probabilities):
        print(f"  {classifier.target_names[i]}: {prob:.4f}")

if __name__ == "__main__":
    main()