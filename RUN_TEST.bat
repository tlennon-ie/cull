@echo off
REM Test harness for 5 images per source

echo ========================================================
echo Pipeline Test: 5 Images Per Source
echo ========================================================

cd /d I:\AI\openclaw\workspace\claude\pipeline_code

echo.
echo Step 1: Running test harness...
python test_harness.py

echo.
echo Step 2: Starting dashboard in background...
start python dashboard_enhanced.py

echo.
echo Step 3: Dashboard running at http://localhost:5000
echo.
echo You can now:
echo   - Check console output
echo   - Visit dashboard
echo   - Monitor queue and sorted folders

pause
