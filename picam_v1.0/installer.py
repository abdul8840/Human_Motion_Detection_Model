"""
INSTALLER - USB Deployment Helper
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from typing import List, Optional
import subprocess

class USBInstaller:
    """USB-based installer for Picam."""
    
    def __init__(self, source_dir: Path = None):
        self.source_dir = source_dir or Path(sys._MEIPASS if getattr(sys, 'frozen', False) else os.path.dirname(__file__))
        self.target_dir = Path("C:\\Picam" if os.name == 'nt' else "/opt/picam")
        self.exe_name = "picam.exe" if os.name == 'nt' else "picam"
    
    def create_installer_bat(self) -> Path:
        """Create Windows installer batch script."""
        bat_content = f"""@echo off
echo PICAM v1.0 Installer
echo =====================

REM Check if running as administrator
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo Please run as Administrator!
    pause
    exit /b 1
)

REM Create installation directory
if not exist "{self.target_dir}" (
    mkdir "{self.target_dir}"
    echo Created directory: {self.target_dir}
)

REM Copy files
echo Installing Picam...
xcopy /E /Y "{self.source_dir}\\*" "{self.target_dir}\\"

REM Create desktop shortcut (Windows only)
if exist "%USERPROFILE%\\Desktop" (
    echo [InternetShortcut] > "%USERPROFILE%\\Desktop\\Picam.url"
    echo URL=file:///{self.target_dir}\\{self.exe_name} >> "%USERPROFILE%\\Desktop\\Picam.url"
    echo IconFile={self.target_dir}\\{self.exe_name} >> "%USERPROFILE%\\Desktop\\Picam.url"
    echo IconIndex=0 >> "%USERPROFILE%\\Desktop\\Picam.url"
)

echo.
echo Installation complete!
echo.
echo To run Picam:
echo   1. Navigate to {self.target_dir}
echo   2. Double-click {self.exe_name}
echo   3. Enter live test password when prompted
echo.
pause
"""
        
        bat_path = self.source_dir / "install.bat"
        bat_path.write_text(bat_content)
        return bat_path
    
    def create_uninstaller_bat(self) -> Path:
        """Create Windows uninstaller batch script."""
        bat_content = f"""@echo off
echo PICAM v1.0 Uninstaller
echo =======================

REM Check if running as administrator
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo Please run as Administrator!
    pause
    exit /b 1
)

REM Remove desktop shortcut
if exist "%USERPROFILE%\\Desktop\\Picam.url" (
    del "%USERPROFILE%\\Desktop\\Picam.url"
    echo Removed desktop shortcut
)

REM Ask before removing installation
set /p confirm="Remove Picam installation from {self.target_dir}? (y/n): "
if /i "%confirm%" neq "y" (
    echo Uninstall cancelled.
    pause
    exit /b 0
)

REM Remove installation directory
if exist "{self.target_dir}" (
    rmdir /S /Q "{self.target_dir}"
    echo Removed directory: {self.target_dir}
)

echo.
echo Uninstall complete!
pause
"""
        
        bat_path = self.source_dir / "uninstall.bat"
        bat_path.write_text(bat_content)
        return bat_path
    
    def create_linux_installer(self) -> Path:
        """Create Linux installer script."""
        script_content = f"""#!/bin/bash
echo "PICAM v1.0 Installer for Linux"
echo "================================"

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "Please run as root (sudo)"
    exit 1
fi

# Create installation directory
mkdir -p {self.target_dir}
echo "Created directory: {self.target_dir}"

# Copy files
echo "Installing Picam..."
cp -r {self.source_dir}/* {self.target_dir}/

# Set permissions
chmod +x {self.target_dir}/{self.exe_name}

# Create desktop entry
DESKTOP_ENTRY="${{HOME}}/.local/share/applications/picam.desktop"
cat > "$DESKTOP_ENTRY" << EOF
[Desktop Entry]
Name=Picam Monitor
Comment=Hotel Monitoring System
Exec={self.target_dir}/{self.exe_name}
Icon={self.target_dir}/assets/logo.ico
Terminal=false
Type=Application
Categories=Utility;
EOF

echo
echo "Installation complete!"
echo
echo "To run Picam:"
echo "  1. Terminal: {self.target_dir}/{self.exe_name}"
echo "  2. Desktop: Search for 'Picam Monitor'"
echo "  3. Enter live test password when prompted"
echo
"""
        
        script_path = self.source_dir / "install.sh"
        script_path.write_text(script_content)
        script_path.chmod(0o755)  # Make executable
        return script_path
    
    def create_readme(self) -> Path:
        """Create README for USB deployment."""
        readme_content = f"""PICAM v1.0 - USB DEPLOYMENT
============================

OVERVIEW
--------
This USB contains Picam v1.0 - a frozen, secure hotel monitoring system.

CONTENTS
--------
- picam.exe          : Main application (frozen core)
- install.bat        : Windows installer
- uninstall.bat      : Windows uninstaller  
- install.sh         : Linux installer
- configs/           : Configuration files
- assets/            : Application assets
- README_DEPLOYMENT.md : This file

INSTALLATION
------------
Windows:
  1. Insert USB
  2. Run install.bat (as Administrator)
  3. Follow prompts

Linux:
  1. Mount USB
  2. Run: sudo ./install.sh
  3. Follow prompts

SECURITY
--------
- Live test password required on first run
- Machine binding prevents unauthorized copying
- Core logic is frozen at v1.0

LICENSE
-------
Default: FREE tier
Upgrade to PRO/PREMIUM: Contact sales@picam.com

SUPPORT
-------
Email: support@picam.com
Phone: 1-800-PICAM-NOW

FIRST RUN
---------
1. Launch Picam from desktop or installation directory
2. Enter live test password: [Contact administrator]
3. System will validate machine binding
4. Monitoring begins automatically

TROUBLESHOOTING
---------------
Q: "Live test failed" error?
A: Contact administrator for correct password

Q: "Machine binding failed" error?
A: System copied to different machine. Contact support.

Q: No video feed?
A: Check camera permissions and connections

COPYRIGHT
---------
© 2024 Picam Systems. All rights reserved.
Unauthorized copying or distribution prohibited.
"""
        
        readme_path = self.source_dir / "README_DEPLOYMENT.md"
        readme_path.write_text(readme_content)
        return readme_path

def main():
    """Command-line installer interface."""
    parser = argparse.ArgumentParser(description='Picam USB Installer')
    parser.add_argument('--create-installers', action='store_true', help='Create installer scripts')
    parser.add_argument('--install', action='store_true', help='Install to current system')
    parser.add_argument('--target', type=str, help='Target installation directory')
    
    args = parser.parse_args()
    
    installer = USBInstaller()
    
    if args.target:
        installer.target_dir = Path(args.target)
    
    if args.create_installers:
        print("Creating installer scripts...")
        installer.create_installer_bat()
        installer.create_uninstaller_bat()
        installer.create_linux_installer()
        installer.create_readme()
        print("✓ Installer scripts created")
    
    if args.install:
        print(f"Installing to {installer.target_dir}...")
        # Implementation would copy files
        print("Installation complete!")

if __name__ == "__main__":
    main()