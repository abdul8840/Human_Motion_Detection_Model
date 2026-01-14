@echo off
echo PICAM v1.0 - Windows Installer
echo ===============================

REM Check administrator privileges
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo ERROR: Please run as Administrator!
    echo Right-click -> Run as Administrator
    pause
    exit /b 1
)

REM Default installation directory
set INSTALL_DIR=C:\Program Files\Picam

echo.
echo Installation Directory: %INSTALL_DIR%
echo.

REM Ask for confirmation
set /p CONFIRM="Install Picam v1.0? (y/n): "
if /i "%CONFIRM%" neq "y" (
    echo Installation cancelled.
    pause
    exit /b 0
)

echo.
echo Installing Picam v1.0...
echo.

REM Create directory structure
if not exist "%INSTALL_DIR%" (
    mkdir "%INSTALL_DIR%"
    echo ✓ Created installation directory
)

REM Copy files
echo Copying files...
xcopy /E /Y /I "%~dp0*" "%INSTALL_DIR%\"

REM Set permissions (optional)
echo Setting permissions...
icacls "%INSTALL_DIR%" /grant Users:(OI)(CI)RX /T >nul 2>&1

REM Create desktop shortcut
if exist "%USERPROFILE%\Desktop" (
    echo [InternetShortcut] > "%USERPROFILE%\Desktop\Picam Monitor.url"
    echo URL=file:///%INSTALL_DIR%\picam.exe >> "%USERPROFILE%\Desktop\Picam Monitor.url"
    echo IconFile=%INSTALL_DIR%\picam.exe >> "%USERPROFILE%\Desktop\Picam Monitor.url"
    echo IconIndex=0 >> "%USERPROFILE%\Desktop\Picam Monitor.url"
    echo ✓ Created desktop shortcut
)

REM Create start menu shortcut
if exist "%ProgramData%\Microsoft\Windows\Start Menu\Programs" (
    mkdir "%ProgramData%\Microsoft\Windows\Start Menu\Programs\Picam" 2>nul
    echo [InternetShortcut] > "%ProgramData%\Microsoft\Windows\Start Menu\Programs\Picam\Picam Monitor.url"
    echo URL=file:///%INSTALL_DIR%\picam.exe >> "%ProgramData%\Microsoft\Windows\Start Menu\Programs\Picam\Picam Monitor.url"
    echo IconFile=%INSTALL_DIR%\picam.exe >> "%ProgramData%\Microsoft\Windows\Start Menu\Programs\Picam\Picam Monitor.url"
    echo IconIndex=0 >> "%ProgramData%\Microsoft\Windows\Start Menu\Programs\Picam\Picam Monitor.url"
    echo ✓ Added to Start Menu
)

echo.
echo ===============================
echo INSTALLATION COMPLETE!
echo ===============================
echo.
echo To run Picam:
echo   1. Double-click "Picam Monitor" on desktop
echo   2. OR Navigate to %INSTALL_DIR% and run picam.exe
echo.
echo On first run:
echo   - Enter live test password (contact administrator)
echo   - System will bind to this machine
echo   - Monitoring will start automatically
echo.
echo Support: support@picam.com
echo.
pause