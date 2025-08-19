import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class EmailSpamClassifier:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.pipeline = None
        self.stemmer = PorterStemmer()
        
        # Download required NLTK data
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess_text(self, text):
        """Clean and preprocess email text"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove punctuation and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove stopwords and stem
        words = text.split()
        words = [self.stemmer.stem(word) for word in words if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(words)
    
    def load_sample_data(self):
        """Create sample spam/ham dataset for demonstration"""
        # Sample spam emails
        spam_emails = [
            "URGENT! You have won $1000000! Click here to claim your prize now!",
            "Congratulations! You've been selected for a special offer. Act now!",
            "FREE MONEY! No strings attached. Send your bank details immediately.",
            "WINNER! You are our lucky winner today. Claim your reward instantly!",
            "Limited time offer! Get rich quick scheme. 100% guaranteed returns!",
            "ALERT: Your account will be suspended. Verify your details now.",
            "Exclusive deal just for you! Make money from home easily!",
            "You've inherited millions! Contact us with your personal information.",
            "URGENT: Update your payment information to avoid account closure.",
            "Amazing opportunity! Earn $5000 per week working from home!"
        ]
        
        # Sample legitimate emails
        ham_emails = [
            "Hi John, hope you're doing well. Let's catch up over coffee this weekend.",
            "Meeting scheduled for tomorrow at 2 PM in conference room A.",
            "Thank you for your purchase. Your order will be delivered in 3-5 days.",
            "Reminder: Your subscription expires next month. Please renew if needed.",
            "Project update: We've completed phase 1 and moving to phase 2.",
            "Happy birthday! Hope you have a wonderful day with family and friends.",
            "The quarterly report is ready for review. Please find it attached.",
            "Weather forecast: Sunny skies expected this weekend. Perfect for outdoor activities.",
            "Your flight booking confirmation for next Tuesday's business trip.",
            "Team lunch at the new restaurant downtown. See you at 12:30 PM."
        ]
        
        # Create DataFrame
        emails = spam_emails + ham_emails
        labels = ['spam'] * len(spam_emails) + ['ham'] * len(ham_emails)
        
        # Add more synthetic data for better training
        additional_spam = [
            "Get viagra cheap! No prescription needed!",
            "Make money fast! Work from home opportunity!",
            "You owe money! Pay now or face consequences!",
            "Free gift card! Click here to redeem!",
            "Lose weight fast! Miracle pill available now!"
        ] * 10
        
        additional_ham = [
            "Looking forward to our meeting next week.",
            "Please review the attached document when you have time.",
            "The weather is nice today, perfect for a walk.",
            "Don't forget about the team building event on Friday.",
            "Your package has been delivered successfully."
        ] * 10
        
        all_emails = emails + additional_spam + additional_ham
        all_labels = labels + ['spam'] * len(additional_spam) + ['ham'] * len(additional_ham)
        
        return pd.DataFrame({'email': all_emails, 'label': all_labels})
    
    def train_model(self, df=None):
        """Train the spam classification model"""
        if df is None:
            df = self.load_sample_data()
        
        print("Preprocessing email data...")
        df['cleaned_email'] = df['email'].apply(self.preprocess_text)
        
        # Prepare features and labels
        X = df['cleaned_email']
        y = df['label']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create pipeline with TF-IDF and Naive Bayes
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('classifier', MultinomialNB(alpha=0.1))
        ])
        
        print("Training model...")
        self.pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.pipeline.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Create confusion matrix plot
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
        plt.title('Confusion Matrix - Email Spam Classification')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig('Task 1/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save the model
        with open('Task 1/spam_classifier_model.pkl', 'wb') as f:
            pickle.dump(self.pipeline, f)
        
        print("Model saved successfully!")
        return accuracy, classification_report(y_test, y_pred, output_dict=True)
    
    def load_model(self):
        """Load the trained model"""
        try:
            with open('Task 1/spam_classifier_model.pkl', 'rb') as f:
                self.pipeline = pickle.load(f)
            return True
        except FileNotFoundError:
            return False
    
    def predict(self, email_text):
        """Predict if an email is spam or not"""
        if self.pipeline is None:
            if not self.load_model():
                raise ValueError("No trained model found. Please train the model first.")
        
        cleaned_text = self.preprocess_text(email_text)
        prediction = self.pipeline.predict([cleaned_text])[0]
        probability = self.pipeline.predict_proba([cleaned_text])[0]
        
        spam_prob = probability[1] if prediction == 'spam' else probability[0]
        
        return {
            'prediction': prediction,
            'confidence': max(probability),
            'spam_probability': spam_prob,
            'ham_probability': 1 - spam_prob
        }

if __name__ == "__main__":
    # Initialize and train the classifier
    classifier = EmailSpamClassifier()
    accuracy, report = classifier.train_model()
    
    # Test with sample emails
    test_emails = [
        "Congratulations! You've won a million dollars! Click here now!",
        "Hi, let's meet for lunch tomorrow at the usual place.",
        "URGENT: Your account needs verification. Send your password now!"
    ]
    
    print("\n" + "="*50)
    print("TESTING THE MODEL")
    print("="*50)
    
    for email in test_emails:
        result = classifier.predict(email)
        print(f"\nEmail: {email[:50]}...")
        print(f"Prediction: {result['prediction'].upper()}")
        print(f"Confidence: {result['confidence']:.4f}")