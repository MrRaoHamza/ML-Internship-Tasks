import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import pickle
import os
import seaborn as sns

class MNISTClassifierSklearn:
    def __init__(self, model_type='mlp'):
        """
        Initialize MNIST classifier with scikit-learn
        model_type: 'mlp' for Neural Network or 'rf' for Random Forest
        """
        self.model_type = model_type
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_and_preprocess_data(self):
        """Load and preprocess MNIST dataset"""
        print("Loading MNIST dataset...")
        
        # Load MNIST data from sklearn
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X, y = mnist.data, mnist.target.astype(int)
        
        # Normalize pixel values to 0-1 range
        X = X / 255.0
        
        # Split into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training data shape: {self.X_train.shape}")
        print(f"Test data shape: {self.X_test.shape}")
        print(f"Classes: {np.unique(y)}")
        
        return (self.X_train, self.y_train), (self.X_test, self.y_test)
    
    def build_model(self):
        """Build model for digit classification"""
        if self.model_type == 'mlp':
            # Multi-layer Perceptron (Neural Network)
            self.model = MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                batch_size='auto',
                learning_rate='constant',
                learning_rate_init=0.001,
                max_iter=300,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                verbose=True
            )
        elif self.model_type == 'rf':
            # Random Forest
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=42,
                n_jobs=-1,
                verbose=1
            )
        
        print(f"Built {self.model_type.upper()} model")
        return self.model
    
    def train_model(self):
        """Train the model"""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        print("Training model...")
        print("This may take several minutes...")
        
        self.model.fit(self.X_train, self.y_train)
        
        print("Training completed!")
        return self.model
    
    def evaluate_model(self):
        """Evaluate model performance"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        print("Evaluating model...")
        
        # Get predictions
        y_pred_train = self.model.predict(self.X_train)
        y_pred_test = self.model.predict(self.X_test)
        
        # Calculate accuracies
        train_accuracy = accuracy_score(self.y_train, y_pred_train)
        test_accuracy = accuracy_score(self.y_test, y_pred_test)
        
        # Generate classification report
        report = classification_report(self.y_test, y_pred_test)
        
        # Generate confusion matrix
        cm = confusion_matrix(self.y_test, y_pred_test)
        
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print("\nClassification Report:")
        print(report)
        
        return {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred_test,
            'true_labels': self.y_test
        }
    
    def plot_confusion_matrix(self, cm):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=range(10), yticklabels=range(10))
        plt.title(f'Confusion Matrix - {self.model_type.upper()} Model')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{self.model_type}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_sample_predictions(self, num_samples=12):
        """Plot sample predictions"""
        # Get random samples
        indices = np.random.choice(len(self.X_test), num_samples, replace=False)
        
        fig, axes = plt.subplots(3, 4, figsize=(12, 9))
        axes = axes.ravel()
        
        for i, idx in enumerate(indices):
            # Reshape image for display
            image = self.X_test[idx].reshape(28, 28)
            true_label = self.y_test[idx]
            pred_label = self.model.predict([self.X_test[idx]])[0]
            
            # Plot image
            axes[i].imshow(image, cmap='gray')
            axes[i].set_title(f'True: {true_label}, Pred: {pred_label}')
            axes[i].axis('off')
            
            # Color border based on correctness
            if true_label == pred_label:
                axes[i].add_patch(plt.Rectangle((0, 0), 27, 27, fill=False, 
                                              edgecolor='green', linewidth=2))
            else:
                axes[i].add_patch(plt.Rectangle((0, 0), 27, 27, fill=False, 
                                              edgecolor='red', linewidth=2))
        
        plt.tight_layout()
        plt.savefig(f'sample_predictions_{self.model_type}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filepath=None):
        """Save the trained model"""
        if filepath is None:
            filepath = f'mnist_model_{self.model_type}.pkl'
        
        if self.model is not None:
            with open(filepath, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"Model saved to {filepath}")
    
    def load_model(self, filepath=None):
        """Load a trained model"""
        if filepath is None:
            filepath = f'mnist_model_{self.model_type}.pkl'
        
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.model = pickle.load(f)
            print(f"Model loaded from {filepath}")
            return True
        return False
    
    def predict_digit(self, image):
        """Predict digit from image"""
        if self.model is None:
            raise ValueError("Model not loaded or trained")
        
        # Ensure image is flattened and normalized
        if len(image.shape) == 2:
            image = image.flatten()
        
        # Normalize if needed
        if image.max() > 1:
            image = image.astype('float32') / 255.0
        
        # Reshape for prediction
        image = image.reshape(1, -1)
        
        # Get prediction and probabilities
        prediction = self.model.predict(image)[0]
        
        # Get probabilities (if available)
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(image)[0]
            confidence = np.max(probabilities)
        else:
            # For models without predict_proba, use a simple confidence measure
            probabilities = np.zeros(10)
            probabilities[prediction] = 1.0
            confidence = 1.0
        
        return int(prediction), float(confidence), probabilities

def main():
    """Main training pipeline"""
    print("MNIST Digit Recognition with Scikit-Learn")
    print("=" * 50)
    
    # Choose model type
    model_type = input("Choose model type (mlp/rf) [mlp]: ").strip().lower()
    if model_type not in ['mlp', 'rf']:
        model_type = 'mlp'
    
    classifier = MNISTClassifierSklearn(model_type=model_type)
    
    # Load and preprocess data
    (X_train, y_train), (X_test, y_test) = classifier.load_and_preprocess_data()
    
    # Build model
    model = classifier.build_model()
    print(f"Model: {model}")
    
    # Train model
    classifier.train_model()
    
    # Evaluate model
    results = classifier.evaluate_model()
    
    # Plot results
    classifier.plot_confusion_matrix(results['confusion_matrix'])
    classifier.plot_sample_predictions()
    
    # Save model and results
    classifier.save_model()
    
    with open(f'evaluation_results_{model_type}.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nTraining completed successfully!")
    print(f"Final Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Model saved as: mnist_model_{model_type}.pkl")

if __name__ == "__main__":
    main()