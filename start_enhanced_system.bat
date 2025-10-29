@echo off
echo ========================================
echo   Smart Investment Planner - Enhanced
echo   Live Data & ML Features
echo ========================================
echo.

echo [1/4] Training Enhanced ML Model...
cd backend
python train_enhanced_model.py
if %errorlevel% neq 0 (
    echo ERROR: Failed to train enhanced model
    pause
    exit /b 1
)
echo.

echo [2/4] Starting Backend Server...
start "Backend Server" cmd /k "python app.py"
timeout /t 3 /nobreak >nul
echo.

echo [3/4] Starting Frontend Server...
cd ..\frontend
start "Frontend Server" cmd /k "npm start"
timeout /t 3 /nobreak >nul
echo.

echo [4/4] Opening Browser...
timeout /t 5 /nobreak >nul
start http://localhost:3000
echo.

echo ========================================
echo   System Started Successfully!
echo ========================================
echo.
echo Features Available:
echo - Live Stock Data Dashboard
echo - Enhanced ML Predictions
echo - Real-time Market Analysis
echo - AI-Powered Investment Advice
echo - Personalized Stock Recommendations
echo.
echo Access the application at: http://localhost:3000
echo.
pause
