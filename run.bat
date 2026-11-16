@echo off
REM Warranty Analytics & Reliability Dashboard Run Script (Windows)

echo ========================================================
echo Starting Warranty Analytics & Reliability Pipeline
echo ========================================================

echo [1/5] Generating Synthetic Data...
python -m src.generate_data
if %errorlevel% neq 0 exit /b %errorlevel%

echo [2/5] Running Data Cleaning & Quality Audit...
python -m src.db
if %errorlevel% neq 0 exit /b %errorlevel%

echo [3/5] Training NLP Failure Mode Classifier...
python -m src.classifier
if %errorlevel% neq 0 exit /b %errorlevel%

echo [4/5] Generating 8D Reports & Excel/Power BI Exports...
python -m src.report
python -m src.export
if %errorlevel% neq 0 exit /b %errorlevel%

echo [5/5] Running Pytest Suite...
python -m pytest tests/ -v
if %errorlevel% neq 0 exit /b %errorlevel%

echo ========================================================
echo All steps completed successfully!
echo Launching Streamlit application...
echo ========================================================
streamlit run app/streamlit_app.py
