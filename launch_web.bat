@echo off
echo ========================================
echo BBB Permeability Web Interface
echo ========================================
echo.
echo Starting Streamlit server...
echo The app will open in your browser at http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

set KMP_DUPLICATE_LIB_OK=TRUE
"C:\Users\nakhi\anaconda3\python.exe" -m streamlit run app.py

pause
