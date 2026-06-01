@echo off
cd /d "%~dp0."

net session >nul 2>&1
if %errorLevel% == 0 (
    netsh advfirewall firewall show rule name="Flask 5000" >nul 2>&1
    if %errorLevel% neq 0 (
        netsh advfirewall firewall add rule name="Flask 5000" dir=in action=allow protocol=tcp localport=5000 >nul 2>&1
        echo [Firewall] Rule added
    ) else (
        echo [Firewall] Rule exists
    )
) else (
    echo [WARN] Not running as admin. Firewall rule skipped.
)

python --version >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] Python not found
    pause
    exit /b 1
)
echo [Python] OK

echo [Server] Starting Flask...
start /b "" python -m web.app

timeout /t 3 >nul
start "" http://localhost:5000

echo ============================================
echo  Server running: http://localhost:5000
echo  Press any key to stop
echo ============================================
pause >nul

echo Stopping server...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":5000" ^| findstr "LISTENING"') do taskkill /PID %%a /F >nul 2>&1
echo Done.
pause
