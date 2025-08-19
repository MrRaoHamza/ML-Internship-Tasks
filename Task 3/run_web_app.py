"""
Web Application Launcher
========================
Simple script to launch the California Housing Price Prediction web app.
"""

import os
import sys
import subprocess
import webbrowser
import time
from threading import Timer

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = ['flask', 'pandas', 'numpy', 'matplotlib', 'seaborn', 'scikit-learn']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Installing missing packages...")
        
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
            print("✅ All packages installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install packages. Please install manually:")
            print(f"   pip install {' '.join(missing_packages)}")
            return False
    
    return True

def open_browser():
    """Open browser after a delay"""
    time.sleep(3)  # Wait for Flask to start
    webbrowser.open('http://localhost:5000')

def main():
    """Main launcher function"""
    print("🏠 California Housing Price Prediction Web App")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        input("Press Enter to exit...")
        return
    
    print("\n🚀 Starting web application...")
    print("📊 Training machine learning model (this may take a moment)...")
    
    # Schedule browser opening
    Timer(3.0, open_browser).start()
    
    try:
        # Import and run the Flask app
        from app import app
        print("\n✅ Web app is running!")
        print("🌐 Open your browser and go to: http://localhost:5000")
        print("⏹️  Press Ctrl+C to stop the server")
        print("-" * 50)
        
        app.run(debug=False, host='0.0.0.0', port=5000)
        
    except ImportError as e:
        print(f"❌ Error importing app: {e}")
        print("Make sure all files are in the correct directory.")
    except KeyboardInterrupt:
        print("\n\n👋 Web app stopped. Thank you for using the Housing Price Predictor!")
    except Exception as e:
        print(f"❌ Error starting web app: {e}")
    
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()