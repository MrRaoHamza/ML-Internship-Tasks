# MNIST Digit Recognition

A complete machine learning solution for recognizing handwritten digits (0-9) using the MNIST dataset. This project includes model training, evaluation, and an interactive web dashboard for testing predictions.

## Features

- **CNN Model**: Convolutional Neural Network optimized for digit recognition
- **Web Dashboard**: Interactive interface for drawing digits and uploading images
- **Real-time Predictions**: Instant digit recognition with confidence scores
- **Model Evaluation**: Comprehensive performance metrics and visualizations
- **Sample Testing**: View predictions on random test samples

## Quick Start

### 1. Install Dependencies
```bash
# Run the installation script
install_dependencies.bat

# Or manually install
pip install -r requirements.txt
```

### 2. Train the Model
```bash
# Run the training script
train_model.bat

# Or manually train
python mnist_classifier.py
```

### 3. Start Web Dashboard
```bash
# Run the web application
run_app.bat

# Or manually start
python app.py
```

Then open your browser to `http://localhost:5000`

## Project Structure

```
Task 2/
├── mnist_classifier.py      # Main training and model code
├── app.py                   # Flask web application
├── requirements.txt         # Python dependencies
├── templates/              # HTML templates
│   ├── base.html           # Base template
│   ├── dashboard.html      # Main dashboard
│   └── train.html          # Training page
├── *.bat                   # Windows batch scripts
└── README.md               # This file
```

## Model Architecture

The CNN model consists of:
- 3 Convolutional layers (32, 64, 64 filters)
- MaxPooling layers for dimensionality reduction
- Dense layers with Dropout for regularization
- Softmax output for 10-class classification

## Expected Performance

- **Training Accuracy**: ~99%
- **Test Accuracy**: ~98-99%
- **Training Time**: 5-15 minutes (depending on hardware)

## Web Dashboard Features

### 1. Model Performance Metrics
- Real-time accuracy and loss display
- Model status indicators

### 2. Interactive Drawing Canvas
- Draw digits with mouse or touch
- Instant prediction with confidence scores
- Probability distribution visualization

### 3. Image Upload
- Upload digit images for prediction
- Supports common image formats (PNG, JPG, etc.)

### 4. Sample Predictions
- View predictions on random test samples
- Compare true labels vs predictions
- Accuracy indicators for each sample

### 5. Probability Visualization
- Bar chart showing prediction probabilities for all digits
- Real-time updates with each prediction

## Usage Examples

### Training the Model
```python
from mnist_classifier import MNISTClassifier

# Create classifier instance
classifier = MNISTClassifier()

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = classifier.load_and_preprocess_data()

# Build and train model
classifier.build_model()
classifier.train_model(x_train, y_train, x_test, y_test, epochs=15)

# Evaluate performance
results = classifier.evaluate_model(x_test, y_test)

# Save model
classifier.save_model()
```

### Making Predictions
```python
# Load trained model
classifier = MNISTClassifier()
classifier.load_model('mnist_model.h5')

# Predict digit from image
predicted_digit, confidence, probabilities = classifier.predict_digit(image_array)
print(f"Predicted digit: {predicted_digit} (confidence: {confidence:.2f})")
```

## API Endpoints

- `GET /` - Main dashboard
- `GET /train` - Training page
- `GET /api/model_info` - Get model performance metrics
- `POST /api/predict` - Make digit predictions
- `GET /api/sample_predictions` - Get random sample predictions
- `POST /api/train` - Start model training

## Requirements

- Python 3.7+
- TensorFlow 2.13+
- Flask 2.3+
- NumPy, Matplotlib, Scikit-learn
- PIL (Pillow) for image processing
- OpenCV for image preprocessing

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Model Not Found**: Train the model first
   ```bash
   python mnist_classifier.py
   ```

3. **Web App Not Starting**: Check if port 5000 is available
   ```bash
   netstat -an | findstr :5000
   ```

4. **Slow Training**: Consider reducing epochs or using GPU acceleration

### Performance Tips

- Use GPU acceleration if available (CUDA-compatible GPU)
- Adjust batch size based on available memory
- Use early stopping to prevent overfitting
- Monitor training progress with validation metrics

## License

This project is for educational purposes. The MNIST dataset is publicly available and free to use.

## Contributing

Feel free to submit issues and enhancement requests!