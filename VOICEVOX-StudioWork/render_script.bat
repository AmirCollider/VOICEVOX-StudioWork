@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1
if "%~1"=="" (
  echo   Drag a script .txt file and drop it on this file. VOICEVOX must be open.
  pause
  exit /b 1
)
python voxtalk.py script "%~1"
echo.
pause
