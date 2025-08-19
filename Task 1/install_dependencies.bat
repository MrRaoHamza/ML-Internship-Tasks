@echo off
echo Installing Email Spam Classification Dependencies...
echo.

echo Step 1: Upgrading pip and setuptools...
python -m pip install --upgrade pip setuptools wheel
echo.

echo Step 2: Installing core dependencies one by one...
python -m pip install flask
python -m pip install numpy
python -m pip install pandas
python -m pip install scikit-learn
python -m pip install nltk
python -m pip install matplotlib
python -m pip install seaborn
python -m pip install Werkzeug
echo.

echo Step 3: Verifying installation...
python -c "import flask, pandas, numpy, sklearn, nltk, matplotlib, seaborn; print('All dependencies installed successfully!')"
echo.

echo Installation complete! You can now run the application.
pause