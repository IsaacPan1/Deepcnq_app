@echo off
setlocal
rem Launch the CNQ app via run.py, preferring the Python launcher.

py -3 --version >nul 2>nul
if %errorlevel%==0 (
    py -3 "%~dp0run.py" %*
    goto :end
)

python --version >nul 2>nul
if %errorlevel%==0 (
    python "%~dp0run.py" %*
    goto :end
)

echo.
echo ERROR: Python was not found on your PATH.
echo Install Python 3.9 or newer from https://www.python.org/downloads/
echo During setup, tick "Add python.exe to PATH", then re-run this file.
echo.
exit /b 1

:end
endlocal
