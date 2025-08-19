@echo off
echo Installing MNIST Digit Recognition dependencies...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.7+ from https://python.org
    pause
    exit /b 1
)

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo Installing required packages...
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo Error installing dependencies. Please check the error messages above.
    pause
    exit /b 1
)

echo.
echo Dependencies installed successfully!
echo.
echo Next steps:
echo 1. Run 'train_model.bat' to train the MNIST model
echo 2. Run 'run_app.bat' to start the web dashboard
echo.
pause