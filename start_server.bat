@echo off
cd /d "%~dp0."

python --version >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] Python not found
    pause
    exit /b 1
)
echo [Python] OK

echo [Server] Starting local Flask service (127.0.0.1 only)...
start /b "" python -m web.app

timeout /t 3 >nul
start "" http://localhost:5000

echo ============================================
echo  Server running: http://localhost:5000
echo  Press any key to stop
echo ============================================
pause >nul

echo Stop the server with Ctrl+C in its console.
pause
