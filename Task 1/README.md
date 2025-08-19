# Email Spam Classification System

A machine learning-powered email spam detection system with an interactive web dashboard.

## Features

- **Advanced Text Preprocessing**: Removes URLs, email addresses, HTML tags, and applies stemming
- **Multiple ML Models**: Uses TF-IDF vectorization with Naive Bayes classifier
- **Interactive Dashboard**: Real-time email classification with confidence scores
- **Model Evaluation**: Confusion matrix and performance metrics visualization
- **Sample Data**: Built-in spam and ham email examples for testing

## Project Structure

```
Task 1/
├── app.py                      # Flask web application
├── spam_classifier.py          # ML model implementation
├── templates/
│   └── dashboard.html         # Web dashboard interface
├── requirements.txt           # Python dependencies
├── README.md                 # This file
├── spam_classifier_model.pkl  # Trained model (generated)
└── confusion_matrix.png      # Model evaluation plot (generated)
```

## Installation & Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   ```bash
   python app.py
   ```

3. **Access Dashboard**:
   Open your browser and go to `http://localhost:5000`

## Usage

### Training the Model

1. Click the "Train Model" button in the dashboard
2. The system will:
   - Load sample spam/ham emails
   - Preprocess the text data
   - Train a Naive Bayes classifier
   - Generate performance metrics
   - Save the trained model

### Classifying Emails

1. Enter email text in the classification panel
2. Click "Classify Email"
3. View the prediction result with confidence scores
4. Use sample emails for quick testing

### Dashboard Features

- **Model Metrics**: Accuracy, Precision, Recall, F1-Score
- **Confusion Matrix**: Visual representation of model performance
- **Sample Emails**: Pre-loaded spam and ham examples
- **Performance Chart**: Classification distribution visualization

## Model Details

### Text Preprocessing Pipeline

1. **Cleaning**: Remove URLs, email addresses, HTML tags
2. **Normalization**: Convert to lowercase, remove punctuation
3. **Tokenization**: Split into individual words
4. **Stopword Removal**: Remove common English stopwords
5. **Stemming**: Reduce words to their root form

### Machine Learning Pipeline

1. **Feature Extraction**: TF-IDF vectorization with n-grams (1,2)
2. **Classification**: Multinomial Naive Bayes with alpha=0.1
3. **Evaluation**: Cross-validation with stratified sampling

### Performance Metrics

The model is evaluated using:
- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

## Sample Results

With the built-in dataset, the model typically achieves:
- Accuracy: ~95%
- Spam Precision: ~90%
- Spam Recall: ~95%
- Ham Precision: ~98%
- Ham Recall: ~95%

## API Endpoints

- `GET /`: Main dashboard page
- `POST /train_model`: Train the classification model
- `POST /predict`: Classify email text
- `GET /get_metrics`: Retrieve model performance metrics
- `GET /get_confusion_matrix`: Get confusion matrix image
- `GET /sample_emails`: Get sample spam/ham emails

## Customization

### Adding Your Own Dataset

Replace the `load_sample_data()` method in `spam_classifier.py` with your own data loading logic:

```python
def load_custom_data(self, csv_file):
    """Load custom dataset from CSV file"""
    df = pd.read_csv(csv_file)
    # Ensure columns are named 'email' and 'label'
    return df
```

### Tuning Model Parameters

Modify the pipeline in the `train_model()` method:

```python
self.pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=10000,  # Increase vocabulary size
        ngram_range=(1, 3),  # Include trigrams
        min_df=2,           # Minimum document frequency
        max_df=0.95         # Maximum document frequency
    )),
    ('classifier', MultinomialNB(alpha=0.01))  # Adjust smoothing
])
```

## Troubleshooting

### Common Issues

1. **NLTK Data Missing**: The app automatically downloads required NLTK data
2. **Model Not Found**: Train the model first using the dashboard
3. **Port Already in Use**: Change the port in `app.py`: `app.run(port=5001)`

### Performance Tips

- For large datasets, consider using `HashingVectorizer` instead of `TfidfVectorizer`
- Implement feature selection to reduce dimensionality
- Try ensemble methods like Random Forest or Gradient Boosting

## Future Enhancements

- [ ] Support for file upload (CSV, EML formats)
- [ ] Real-time email monitoring
- [ ] Advanced feature engineering (email headers, metadata)
- [ ] Deep learning models (LSTM, BERT)
- [ ] Multi-language support
- [ ] Email attachment analysis

## License

This project is open source and available under the MIT License.