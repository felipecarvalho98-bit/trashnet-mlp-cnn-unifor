@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv_gpu\Scripts\python.exe" (
  echo Execute PREPARAR_AMBIENTE.bat primeiro.
  pause
  exit /b 1
)
".venv_gpu\Scripts\python.exe" -m src.train --epochs 60
if errorlevel 1 (
  echo.
  echo O treinamento terminou com erro. Confira a mensagem acima e a pasta outputs.
  pause
  exit /b 1
)
echo.
echo Treinamento finalizado. Os arquivos estao na pasta outputs.
pause
