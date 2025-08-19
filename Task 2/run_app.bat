@echo off
echo Starting MNIST Digit Recognition Web Dashboard...
echo.

REM Check if model exists
if not exist "mnist_model.h5" (
    echo Warning: No trained model found!
    echo Please run 'train_model.bat' first to train the model.
    echo.
    echo You can still run the web app, but predictions won't work.
    echo.
    pause
)

echo Starting Flask web server...
echo The dashboard will be available at: http://localhost:5000
echo Press Ctrl+C to stop the server
echo.

python app.py

pause