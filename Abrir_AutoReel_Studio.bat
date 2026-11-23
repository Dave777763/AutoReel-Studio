@echo off
title AutoReel Studio - 9:16 Vertical Video Builder
cd /d "%~dp0"
echo ========================================================
echo   Iniciando AutoReel Studio (1080x1920 @ 30fps)
echo ========================================================
echo.
python -m streamlit run app.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] No se pudo iniciar la aplicacion.
    pause
)
