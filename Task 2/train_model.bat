@echo off
echo Training MNIST Digit Recognition Model...
echo.
echo This will train a CNN model on the MNIST dataset.
echo Training may take 5-15 minutes depending on your hardware.
echo.
pause

python mnist_classifier.py

if errorlevel 1 (
    echo.
    echo Error during training. Please check the error messages above.
    pause
    exit /b 1
)

echo.
echo Training completed successfully!
echo Model saved as 'mnist_model.h5'
echo You can now run the web dashboard with 'run_app.bat'
echo.
pause