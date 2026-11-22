@echo off
setlocal
cd /d "%~dp0"
echo ============================================================
echo AMBIENTE ISOLADO - PYTORCH + DIRECTML (AMD)
echo ============================================================
where py >nul 2>nul
if errorlevel 1 (
  echo ERRO: o Python Launcher 'py' nao foi encontrado.
  echo Instale o Python 3.12 de 64 bits e marque a opcao Python Launcher.
  pause
  exit /b 1
)
py -3.12 -c "import sys; print(sys.version)" >nul 2>nul
if errorlevel 1 (
  echo ERRO: Python 3.12 nao foi encontrado.
  echo Este ambiente DirectML usa o Python 3.12 de 64 bits, sem alterar seu Python 3.13 principal.
  pause
  exit /b 1
)
if not exist ".venv_gpu\Scripts\python.exe" (
  echo Criando ambiente virtual isolado...
  py -3.12 -m venv .venv_gpu
  if errorlevel 1 goto erro
)
echo Atualizando pip...
".venv_gpu\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto erro
echo Instalando PyTorch DirectML e bibliotecas do projeto...
".venv_gpu\Scripts\python.exe" -m pip install -r requirements-directml.txt
if errorlevel 1 goto erro
echo.
echo AMBIENTE PREPARADO.
echo Agora execute CHECK_GPU.bat antes do treinamento completo.
pause
exit /b 0
:erro
echo.
echo ERRO ao preparar o ambiente. Leia a mensagem acima.
pause
exit /b 1
