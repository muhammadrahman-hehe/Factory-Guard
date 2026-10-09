@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" python -m venv --system-site-packages .venv
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
  pause
  exit /b 1
)
if not exist "models\forklift.pt" goto models
if not exist "models\ppe.pt" goto models
if not exist "models\ppe_full.pt" goto models
if not exist "models\fire.pt" goto models
if not exist "models\fall.pt" goto models
goto run
:models
".venv\Scripts\python.exe" setup_models.py
if errorlevel 1 (
  pause
  exit /b 1
)
:run
".venv\Scripts\python.exe" -m streamlit run app.py --server.address 127.0.0.1
pause
