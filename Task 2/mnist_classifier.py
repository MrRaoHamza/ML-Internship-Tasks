import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os

class MNISTClassifier:
    def __init__(self):
        self.model = None
        self.history = None
        
    def load_and_preprocess_data(self):
        """Load and preprocess MNIST dataset"""
        print("Loading MNIST dataset...")
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        
        # Normalize pixel values to 0-1 range
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Reshape data to add channel dimension
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        
        # Convert labels to categorical
        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)
        
        print(f"Training data shape: {x_train.shape}")
        print(f"Test data shape: {x_test.shape}")
        
        return (x_train, y_train), (x_test, y_test)
    
    def build_model(self):
        """Build CNN model for digit classification"""
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train_model(self, x_train, y_train, x_test, y_test, epochs=10):
        """Train the model"""
        print("Training model...")
        
        # Add callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=2)
        ]
        
        self.history = self.model.fit(
            x_train, y_train,
            batch_size=128,
            epochs=epochs,
            validation_data=(x_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def evaluate_model(self, x_test, y_test):
        """Evaluate model performance"""
        print("Evaluating model...")
        
        # Get predictions
        y_pred = self.model.predict(x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # Calculate accuracy
        test_loss, test_accuracy = self.model.evaluate(x_test, y_test, verbose=0)
        
        # Generate classification report
        report = classification_report(y_true_classes, y_pred_classes)
        
        # Generate confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print("\nClassification Report:")
        print(report)
        
        return {
            'accuracy': test_accuracy,
            'loss': test_loss,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred_classes,
            'true_labels': y_true_classes
        }
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('Task 2/training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, cm):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        tick_marks = np.arange(10)
        plt.xticks(tick_marks, range(10))
        plt.yticks(tick_marks, range(10))
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(10):
            for j in range(10):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('Task 2/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filepath='Task 2/mnist_model.h5'):
        """Save the trained model"""
        if self.model is not None:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='Task 2/mnist_model.h5'):
        """Load a trained model"""
        if os.path.exists(filepath):
            self.model = keras.models.load_model(filepath)
            print(f"Model loaded from {filepath}")
            return True
        return False
    
    def predict_digit(self, image):
        """Predict digit from image"""
        if self.model is None:
            raise ValueError("Model not loaded or trained")
        
        # Ensure image is in correct format
        if len(image.shape) == 2:
            image = image.reshape(1, 28, 28, 1)
        elif len(image.shape) == 3:
            image = image.reshape(1, 28, 28, 1)
        
        # Normalize if needed
        if image.max() > 1:
            image = image.astype('float32') / 255.0
        
        prediction = self.model.predict(image)
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction)
        
        return predicted_digit, confidence, prediction[0]

def main():
    """Main training pipeline"""
    classifier = MNISTClassifier()
    
    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = classifier.load_and_preprocess_data()
    
    # Build model
    model = classifier.build_model()
    print(model.summary())
    
    # Train model
    history = classifier.train_model(x_train, y_train, x_test, y_test, epochs=15)
    
    # Evaluate model
    results = classifier.evaluate_model(x_test, y_test)
    
    # Plot results
    classifier.plot_training_history()
    classifier.plot_confusion_matrix(results['confusion_matrix'])
    
    # Save model
    classifier.save_model()
    
    # Save evaluation results
    with open('Task 2/evaluation_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()