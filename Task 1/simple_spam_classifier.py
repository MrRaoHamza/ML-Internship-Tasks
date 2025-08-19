"""
Simplified Email Spam Classifier - Minimal Dependencies Version
This version uses only basic libraries to avoid compatibility issues
"""

import re
import string
import pickle
import json
from collections import Counter
import math

class SimpleEmailSpamClassifier:
    def __init__(self):
        self.spam_words = {}
        self.ham_words = {}
        self.spam_count = 0
        self.ham_count = 0
        self.vocabulary = set()
        
    def preprocess_text(self, text):
        """Simple text preprocessing"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs and email addresses
        text = re.sub(r'http\S+|www\S+|https\S+|\S+@\S+', '', text)
        
        # Remove punctuation and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Split into words and filter
        words = text.split()
        words = [word for word in words if len(word) > 2]
        
        return ' '.join(words)
    
    def get_sample_data(self):
        """Generate sample training data"""
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
            "Amazing opportunity! Earn $5000 per week working from home!",
            "Get viagra cheap! No prescription needed!",
            "Make money fast! Work from home opportunity!",
            "You owe money! Pay now or face consequences!",
            "Free gift card! Click here to redeem!",
            "Lose weight fast! Miracle pill available now!"
        ] * 5  # Multiply for more training data
        
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
            "Team lunch at the new restaurant downtown. See you at 12:30 PM.",
            "Looking forward to our meeting next week about the new project.",
            "Please review the attached document when you have time.",
            "The weather is nice today, perfect for a walk in the park.",
            "Don't forget about the team building event on Friday afternoon.",
            "Your package has been delivered successfully to your address."
        ] * 5  # Multiply for more training data
        
        return spam_emails, ham_emails
    
    def train(self):
        """Train the Naive Bayes classifier"""
        spam_emails, ham_emails = self.get_sample_data()
        
        print("Training simple spam classifier...")
        
        # Process spam emails
        for email in spam_emails:
            words = self.preprocess_text(email).split()
            for word in words:
                self.spam_words[word] = self.spam_words.get(word, 0) + 1
                self.vocabulary.add(word)
            self.spam_count += 1
        
        # Process ham emails
        for email in ham_emails:
            words = self.preprocess_text(email).split()
            for word in words:
                self.ham_words[word] = self.ham_words.get(word, 0) + 1
                self.vocabulary.add(word)
            self.ham_count += 1
        
        print(f"Training completed!")
        print(f"Spam emails: {self.spam_count}")
        print(f"Ham emails: {self.ham_count}")
        print(f"Vocabulary size: {len(self.vocabulary)}")
        
        # Save model
        self.save_model()
        
        # Test the model
        self.test_model()
    
    def calculate_probability(self, words, word_counts, total_count):
        """Calculate probability using Naive Bayes"""
        prob = 0
        vocab_size = len(self.vocabulary)
        
        for word in words:
            word_count = word_counts.get(word, 0)
            # Laplace smoothing
            word_prob = (word_count + 1) / (sum(word_counts.values()) + vocab_size)
            prob += math.log(word_prob)
        
        # Add class probability
        class_prob = total_count / (self.spam_count + self.ham_count)
        prob += math.log(class_prob)
        
        return prob
    
    def predict(self, email_text):
        """Predict if email is spam or ham"""
        words = self.preprocess_text(email_text).split()
        
        if not words:
            return {
                'prediction': 'ham',
                'confidence': 0.5,
                'spam_probability': 0.5,
                'ham_probability': 0.5
            }
        
        # Calculate probabilities
        spam_prob = self.calculate_probability(words, self.spam_words, self.spam_count)
        ham_prob = self.calculate_probability(words, self.ham_words, self.ham_count)
        
        # Determine prediction
        if spam_prob > ham_prob:
            prediction = 'spam'
            confidence = 1 / (1 + math.exp(ham_prob - spam_prob))
        else:
            prediction = 'ham'
            confidence = 1 / (1 + math.exp(spam_prob - ham_prob))
        
        spam_probability = 1 / (1 + math.exp(ham_prob - spam_prob))
        ham_probability = 1 - spam_probability
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'spam_probability': spam_probability,
            'ham_probability': ham_probability
        }
    
    def test_model(self):
        """Test the model with sample emails"""
        test_emails = [
            ("URGENT! You've won money! Click now!", "spam"),
            ("Hi, let's meet for lunch tomorrow.", "ham"),
            ("Free money! Send your bank details!", "spam"),
            ("Meeting at 3 PM in conference room.", "ham"),
            ("Get rich quick! Amazing opportunity!", "spam")
        ]
        
        correct = 0
        total = len(test_emails)
        
        print("\nTesting model:")
        print("-" * 50)
        
        for email, expected in test_emails:
            result = self.predict(email)
            predicted = result['prediction']
            confidence = result['confidence']
            
            is_correct = predicted == expected
            if is_correct:
                correct += 1
            
            status = "✓" if is_correct else "✗"
            print(f"{status} {email[:30]}... -> {predicted.upper()} ({confidence:.2f})")
        
        accuracy = correct / total
        print(f"\nAccuracy: {accuracy:.2f} ({correct}/{total})")
        
        return accuracy
    
    def save_model(self):
        """Save the trained model"""
        model_data = {
            'spam_words': self.spam_words,
            'ham_words': self.ham_words,
            'spam_count': self.spam_count,
            'ham_count': self.ham_count,
            'vocabulary': list(self.vocabulary)
        }
        
        with open('simple_model.json', 'w') as f:
            json.dump(model_data, f)
        
        print("Model saved to simple_model.json")
    
    def load_model(self):
        """Load the trained model"""
        try:
            with open('simple_model.json', 'r') as f:
                model_data = json.load(f)
            
            self.spam_words = model_data['spam_words']
            self.ham_words = model_data['ham_words']
            self.spam_count = model_data['spam_count']
            self.ham_count = model_data['ham_count']
            self.vocabulary = set(model_data['vocabulary'])
            
            return True
        except FileNotFoundError:
            return False

if __name__ == "__main__":
    # Initialize and train classifier
    classifier = SimpleEmailSpamClassifier()
    
    # Try to load existing model, otherwise train new one
    if not classifier.load_model():
        classifier.train()
    else:
        print("Loaded existing model")
        classifier.test_model()
    
    # Interactive testing
    print("\n" + "="*50)
    print("INTERACTIVE EMAIL TESTING")
    print("="*50)
    print("Enter email text to classify (or 'quit' to exit):")
    
    while True:
        email = input("\nEmail: ").strip()
        if email.lower() == 'quit':
            break
        
        if email:
            result = classifier.predict(email)
            print(f"Prediction: {result['prediction'].upper()}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Spam probability: {result['spam_probability']:.4f}")
            print(f"Ham probability: {result['ham_probability']:.4f}")