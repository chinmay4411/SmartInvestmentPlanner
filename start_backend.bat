@echo off
echo Starting Smart Investment Planner Backend...
cd backend
python setup.py
if %errorlevel% neq 0 (
    echo Setup failed!
    pause
    exit /b 1
)
echo.
echo Starting Flask server...
python app.py
pause
