@echo off
echo ========================================
echo Smart Investment Planner - Full System
echo ========================================
echo.

echo Starting Backend Server...
start "Backend Server" cmd /k "cd backend && python app.py"

echo Waiting for backend to start...
timeout /t 5 /nobreak > nul

echo Starting Frontend Server...
start "Frontend Server" cmd /k "cd frontend && npm start"

echo.
echo ========================================
echo System Started Successfully!
echo ========================================
echo.
echo Backend: http://127.0.0.1:5500
echo Frontend: http://localhost:3000
echo.
echo Features Available:
echo - Investment Portfolio Prediction
echo - AI-Powered Financial Advice (Local LLM)
echo - Stock Price Prediction
echo - Interactive Analytics Dashboard
echo - RAG-based Knowledge System
echo.
echo Press any key to exit...
pause > nul
