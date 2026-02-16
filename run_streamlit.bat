@echo off
echo ============================================
echo   AI Shoplifting Detection Dashboard
echo ============================================
echo.

cd /d "%~dp0"
call venv\Scripts\activate

echo Starting Streamlit...
echo Browser will open at http://localhost:8501
echo Press Ctrl+C to stop.
echo.

streamlit run src\dashboard\video_detection_app.py --server.port 8501
pause
