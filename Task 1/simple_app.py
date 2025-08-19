"""
Simple Flask App for Email Spam Classification
Minimal dependencies version that works with Python 3.13
"""

try:
    from flask import Flask, render_template_string, request, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Flask not available. Please install: pip install flask")

import json
import os
from simple_spam_classifier import SimpleEmailSpamClassifier

# Simple HTML template embedded in Python
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Simple Email Spam Classifier</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            color: #333;
            margin-bottom: 30px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #555;
        }
        textarea {
            width: 100%;
            padding: 10px;
            border: 2px solid #ddd;
            border-radius: 5px;
            font-size: 14px;
            resize: vertical;
        }
        .btn {
            background: #007bff;
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            margin-right: 10px;
        }
        .btn:hover {
            background: #0056b3;
        }
        .btn-secondary {
            background: #6c757d;
        }
        .btn-secondary:hover {
            background: #545b62;
        }
        .result {
            margin-top: 20px;
            padding: 15px;
            border-radius: 5px;
            font-weight: bold;
        }
        .spam {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .ham {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .metric-card {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
            border: 1px solid #dee2e6;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #007bff;
        }
        .samples {
            margin-top: 30px;
        }
        .sample-email {
            background: #f8f9fa;
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
            cursor: pointer;
            border: 1px solid #dee2e6;
        }
        .sample-email:hover {
            background: #e9ecef;
        }
        .loading {
            display: none;
            text-align: center;
            color: #007bff;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛡️ Email Spam Classifier</h1>
            <p>Simple machine learning-powered spam detection</p>
        </div>

        <div class="metrics" id="metrics">
            <div class="metric-card">
                <div class="metric-value" id="accuracy">--</div>
                <div>Accuracy</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="total-emails">--</div>
                <div>Total Emails</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="vocab-size">--</div>
                <div>Vocabulary</div>
            </div>
        </div>

        <button class="btn" onclick="trainModel()">🔄 Train Model</button>
        <button class="btn btn-secondary" onclick="loadMetrics()">📊 Load Metrics</button>

        <div class="form-group" style="margin-top: 30px;">
            <label for="emailText">Enter Email Text:</label>
            <textarea id="emailText" rows="6" placeholder="Paste your email content here..."></textarea>
        </div>

        <button class="btn" onclick="classifyEmail()">🔍 Classify Email</button>
        <button class="btn btn-secondary" onclick="clearText()">🗑️ Clear</button>

        <div class="loading" id="loading">
            <p>Processing...</p>
        </div>

        <div id="result"></div>

        <div class="samples">
            <h3>📧 Sample Emails (Click to Test)</h3>
            <div>
                <h4 style="color: #dc3545;">Spam Examples:</h4>
                <div class="sample-email" onclick="useEmail(this)">
                    URGENT! You have won $1000000! Click here to claim your prize now!
                </div>
                <div class="sample-email" onclick="useEmail(this)">
                    FREE MONEY! No strings attached. Send your bank details immediately.
                </div>
                <div class="sample-email" onclick="useEmail(this)">
                    Get viagra cheap! No prescription needed!
                </div>
            </div>
            <div style="margin-top: 20px;">
                <h4 style="color: #28a745;">Ham Examples:</h4>
                <div class="sample-email" onclick="useEmail(this)">
                    Hi John, hope you're doing well. Let's catch up over coffee this weekend.
                </div>
                <div class="sample-email" onclick="useEmail(this)">
                    Meeting scheduled for tomorrow at 2 PM in conference room A.
                </div>
                <div class="sample-email" onclick="useEmail(this)">
                    Thank you for your purchase. Your order will be delivered in 3-5 days.
                </div>
            </div>
        </div>
    </div>

    <script>
        function trainModel() {
            const btn = event.target;
            const originalText = btn.innerHTML;
            btn.innerHTML = '⏳ Training...';
            btn.disabled = true;
            
            showLoading(true);

            fetch('/train', {method: 'POST'})
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert('✅ Model trained successfully!\\nAccuracy: ' + (data.accuracy * 100).toFixed(1) + '%');
                    loadMetrics();
                } else {
                    alert('❌ Error: ' + data.message);
                }
            })
            .catch(error => {
                alert('❌ Error: ' + error.message);
            })
            .finally(() => {
                btn.innerHTML = originalText;
                btn.disabled = false;
                showLoading(false);
            });
        }

        function classifyEmail() {
            const emailText = document.getElementById('emailText').value.trim();
            if (!emailText) {
                alert('⚠️ Please enter an email text to classify.');
                return;
            }

            showLoading(true);

            fetch('/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({email_text: emailText})
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showResult(data);
                } else {
                    alert('❌ Error: ' + data.message);
                }
            })
            .catch(error => {
                alert('❌ Error: ' + error.message);
            })
            .finally(() => {
                showLoading(false);
            });
        }

        function showResult(data) {
            const resultDiv = document.getElementById('result');
            const isSpam = data.prediction === 'spam';
            const resultClass = isSpam ? 'spam' : 'ham';
            const icon = isSpam ? '⚠️' : '✅';
            
            resultDiv.innerHTML = `
                <div class="result ${resultClass}">
                    <h4>${icon} Classification Result</h4>
                    <p><strong>Prediction:</strong> ${data.prediction.toUpperCase()}</p>
                    <p><strong>Confidence:</strong> ${(data.confidence * 100).toFixed(2)}%</p>
                    <p><strong>Spam Probability:</strong> ${(data.spam_probability * 100).toFixed(2)}%</p>
                    <p><strong>Ham Probability:</strong> ${(data.ham_probability * 100).toFixed(2)}%</p>
                </div>
            `;
        }

        function loadMetrics() {
            fetch('/metrics')
            .then(response => response.json())
            .then(data => {
                if (data.accuracy !== undefined) {
                    document.getElementById('accuracy').textContent = (data.accuracy * 100).toFixed(1) + '%';
                    document.getElementById('total-emails').textContent = data.total_emails;
                    document.getElementById('vocab-size').textContent = data.vocab_size;
                }
            });
        }

        function useEmail(element) {
            document.getElementById('emailText').value = element.textContent.trim();
        }

        function clearText() {
            document.getElementById('emailText').value = '';
            document.getElementById('result').innerHTML = '';
        }

        function showLoading(show) {
            document.getElementById('loading').style.display = show ? 'block' : 'none';
        }

        // Load metrics on page load
        window.onload = function() {
            loadMetrics();
        };
    </script>
</body>
</html>
"""

if FLASK_AVAILABLE:
    app = Flask(__name__)
    classifier = SimpleEmailSpamClassifier()

    @app.route('/')
    def dashboard():
        return render_template_string(HTML_TEMPLATE)

    @app.route('/train', methods=['POST'])
    def train_model():
        try:
            accuracy = classifier.train()
            return jsonify({
                'success': True,
                'message': 'Model trained successfully!',
                'accuracy': accuracy
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'message': str(e)
            })

    @app.route('/predict', methods=['POST'])
    def predict_email():
        try:
            data = request.get_json()
            email_text = data.get('email_text', '')
            
            if not email_text.strip():
                return jsonify({
                    'success': False,
                    'message': 'Please enter an email text to classify.'
                })
            
            result = classifier.predict(email_text)
            
            return jsonify({
                'success': True,
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'spam_probability': result['spam_probability'],
                'ham_probability': result['ham_probability']
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'message': str(e)
            })

    @app.route('/metrics')
    def get_metrics():
        try:
            total_emails = classifier.spam_count + classifier.ham_count
            vocab_size = len(classifier.vocabulary)
            
            # Calculate accuracy if model is trained
            accuracy = 0
            if total_emails > 0:
                accuracy = classifier.test_model()
            
            return jsonify({
                'accuracy': accuracy,
                'total_emails': total_emails,
                'vocab_size': vocab_size
            })
        except:
            return jsonify({
                'accuracy': 0,
                'total_emails': 0,
                'vocab_size': 0
            })

def run_simple_server():
    """Run a simple HTTP server without Flask"""
    import http.server
    import socketserver
    import webbrowser
    import threading
    
    class SimpleHandler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/':
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(HTML_TEMPLATE.encode())
            else:
                super().do_GET()
    
    PORT = 8000
    with socketserver.TCPServer(("", PORT), SimpleHandler) as httpd:
        print(f"Simple server running at http://localhost:{PORT}")
        print("Note: This version has limited functionality without Flask")
        webbrowser.open(f'http://localhost:{PORT}')
        httpd.serve_forever()

if __name__ == '__main__':
    # Try to load existing model
    if classifier.load_model():
        print("✅ Existing model loaded successfully!")
    else:
        print("ℹ️ No existing model found. Training new model...")
        classifier.train()
    
    if FLASK_AVAILABLE:
        print("🚀 Starting Flask application...")
        print("📱 Open your browser and go to: http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("⚠️ Flask not available. Starting simple server...")
        run_simple_server()