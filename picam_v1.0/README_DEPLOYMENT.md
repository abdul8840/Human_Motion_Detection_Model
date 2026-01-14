# PICAM v1.0 - Deployment Guide

## QUICK START
1. **Build EXE**: Run `python build_exe.py`
2. **Copy to USB**: Copy `usb_deployment/` folder to USB drive
3. **Install**: On target machine, run `install.bat` (Windows) or `./install.sh` (Linux)
4. **Run**: Launch Picam, enter live test password

## SECURITY FEATURES

### 1. Live Test Protection
- Password required on first run
- Default: `PicamLiveTest2024!` (CHANGE BEFORE DEPLOYMENT!)
- 3 attempts allowed before lockout
- Change password in `build_exe.py`

### 2. Machine Binding
- EXE binds to first machine it runs on
- Prevents copying to other machines
- Fallback mode available for virtual machines

### 3. Frozen Core Logic
- Core detection logic cannot be modified
- All changes go through version control
- Checksum validation on startup

## BUILD INSTRUCTIONS

### Prerequisites
```bash
pip install pyinstaller numpy opencv-python pillow