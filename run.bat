@echo off
echo ==========================================
echo   TINNAM AI DATA SYSTEM
echo ==========================================
echo.

:: Check Python
python --version 2>nul || (
    echo Python is not installed! Please install Python 3.8+.
    pause
    exit /b 1
)

:: Install dependencies
echo [1/3] Installing dependencies...
pip install -r requirements.txt -q

:: Optional: Install TensorFlow for deep learning
echo.
echo [Optional] Install TensorFlow for LSTM/Transformer models?
echo (Without TensorFlow, models will use fallback prediction)
set /p tf_choice="Install TensorFlow? (y/n): "
if /i "%tf_choice%"=="y" (
    echo Installing TensorFlow...
    pip install tensorflow -q
)

:: Run the application
echo.
echo [2/3] Starting TinNam AI...
echo Open http://localhost:5000 in your browser
echo.
cd /d E:\TinNamAI\app
python app.py
pause
