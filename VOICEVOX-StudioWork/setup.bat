@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1
echo   Installing what VoxTalk needs (numpy + e2k, about 25 MB, once)...
python -m pip install -r requirements.txt || (echo. & echo   Install failed. Is Python installed and on PATH? & pause & exit /b 1)
echo.
echo   Done. Now: open VOICEVOX, then drag a script .txt onto render_script.bat
pause
