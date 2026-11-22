@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv_gpu\Scripts\python.exe" (
  echo Ambiente Python nao encontrado. Execute PREPARAR_AMBIENTE.bat primeiro.
  pause
  exit /b 1
)
".venv_gpu\Scripts\python.exe" -m src.backup_results
pause
