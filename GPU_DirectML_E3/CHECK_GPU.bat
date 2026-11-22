@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv_gpu\Scripts\python.exe" (
  echo Execute PREPARAR_AMBIENTE.bat primeiro.
  pause
  exit /b 1
)
".venv_gpu\Scripts\python.exe" -m src.train --check-only
if errorlevel 1 (
  echo.
  echo O teste nao foi aprovado. O treinamento completo nao foi iniciado.
  pause
  exit /b 1
)
echo.
echo Teste aprovado. A CNN E3 completa e os contadores de GPU foram verificados.
pause
