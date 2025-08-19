@echo off
echo Starting Iris Classification Dashboard...
echo.

echo Step 1: Training the machine learning model...
python iris_classifier.py

echo.
echo Step 2: Starting the web dashboard...
echo Dashboard will be available at: http://localhost:5000
echo.

python app.py

pause