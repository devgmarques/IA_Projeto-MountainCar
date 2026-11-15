@echo off
cd /d "%~dp0"
py -3.12 --version
if errorlevel 1 goto python_missing
if not exist ".venv\Scripts\python.exe" py -3.12 -m venv .venv
if errorlevel 1 goto failed
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto failed
.venv\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto failed
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto failed
.venv\Scripts\python.exe verificar.py
if errorlevel 1 goto failed
echo Instalacao concluida. Leia GUIA_COMPLETO.md para treinar e apresentar.
pause
exit /b 0
:python_missing
echo Instale o Python 3.12 de 64 bits conforme o guia. Nao desinstale seu outro Python.
pause
exit /b 1
:failed
echo A instalacao falhou. Copie o erro acima e confira o guia.
pause
exit /b 1
