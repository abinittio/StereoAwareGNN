@echo off
cls
color 0A
echo.
echo ========================================================================
echo                  BBB PERMEABILITY WEB INTERFACE
echo ========================================================================
echo.
echo  Starting the beautiful web interface...
echo.
echo  The app will automatically open in your browser at:
echo  http://localhost:8501
echo.
echo  Features:
echo  - Beautiful interactive UI with gradients
echo  - 20+ pre-loaded molecules to test
echo  - Real-time predictions
echo  - Interactive charts and visualizations
echo  - Export results to CSV/JSON
echo.
echo ========================================================================
echo.
echo  Press Ctrl+C to stop the server
echo.
echo ========================================================================
echo.

set KMP_DUPLICATE_LIB_OK=TRUE
cd /d "%~dp0"
start http://localhost:8501
"C:\Users\nakhi\anaconda3\python.exe" -m streamlit run app.py

pause
