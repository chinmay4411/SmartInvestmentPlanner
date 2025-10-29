@echo off
echo Starting Smart Investment Planner Frontend...
cd frontend
npm install
if %errorlevel% neq 0 (
    echo npm install failed!
    pause
    exit /b 1
)
echo.
echo Starting React development server...
npm start
pause
