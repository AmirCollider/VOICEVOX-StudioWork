@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1
set /p T=Type an English line: 
python voxtalk.py kana "%T%"
echo.
python voxtalk.py say "%T%"
echo   (renders\say.wav)
pause
