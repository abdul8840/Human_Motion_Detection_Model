"""
PYINSTALLER BUILD SCRIPT - Create frozen EXE
"""

import os
import sys
import json
import hashlib
import shutil
from pathlib import Path
import PyInstaller.__main__

def generate_password_hash(password: str) -> str:
    """Generate password hash for security config."""
    salt = "PICAM_v1.0_SALT_2024"
    return hashlib.sha256((password + salt).encode()).hexdigest()

def create_security_config(password: str = None):
    """Create security configuration with hashed password."""
    if password is None:
        # Default password - CHANGE THIS IN PRODUCTION!
        password = "PicamLiveTest2024!"
        print(f"⚠️  Using default password: {password}")
        print("⚠️  CHANGE THIS IN PRODUCTION!")
    
    password_hash = generate_password_hash(password)
    
    config = {
        "require_live_test": True,
        "require_machine_binding": True,
        "allow_fallback": True,
        "max_password_attempts": 3,
        "lockout_time_minutes": 5,
        "password_hash": password_hash
    }
    
    config_dir = Path("configs")
    config_dir.mkdir(exist_ok=True)
    
    config_file = config_dir / "security_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Security config created: {config_file}")
    print(f"  Password hash: {password_hash[:16]}...")
    
    return config_file

def build_exe(onefile: bool = True, console: bool = True, optimize: bool = True):
    """Build Picam EXE with PyInstaller."""
    
    # Create dist directory
    dist_dir = Path("dist")
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    dist_dir.mkdir()
    
    # Create USB deployment directory
    usb_dir = Path("usb_deployment")
    if usb_dir.exists():
        shutil.rmtree(usb_dir)
    usb_dir.mkdir()
    
    # PyInstaller arguments
    args = [
        'main.py',
        '--name=picam',
        f'--distpath={dist_dir}',
        '--workpath=build',
        '--specpath=build'
    ]
    
    if onefile:
        args.append('--onefile')
    
    if not console:
        args.append('--windowed')
    
    if optimize:
        args.extend(['--optimize=2'])
    
    # Add data files
    data_files = [
        ('configs', 'configs'),
        ('assets', 'assets')
    ]
    
    for src, dest in data_files:
        if Path(src).exists():
            args.extend(['--add-data', f'{src}{os.pathsep}{dest}'])
    
    # Add hidden imports if needed
    hidden_imports = [
        'numpy',
        'opencv-python',
        'PIL'
    ]
    
    for imp in hidden_imports:
        args.extend(['--hidden-import', imp])
    
    # Windows specific
    if os.name == 'nt':
        args.extend([
            '--icon=assets/logo.ico',
            '--uac-admin'  # Request admin privileges
        ])
    
    # Linux specific
    if os.name == 'posix':
        args.extend([
            '--strip',
            '--runtime-hook=rthook.py'
        ])
    
    print("Building EXE with PyInstaller...")
    print(f"Command: pyinstaller {' '.join(args)}")
    
    try:
        PyInstaller.__main__.run(args)
        print("✓ EXE build complete!")
    except Exception as e:
        print(f"❌ Build failed: {e}")
        return False
    
    # Copy to USB deployment directory
    exe_name = "picam.exe" if os.name == 'nt' else "picam"
    exe_src = dist_dir / exe_name
    
    if exe_src.exists():
        # Copy EXE
        shutil.copy2(exe_src, usb_dir / exe_name)
        
        # Copy configs and assets
        for src, dest in data_files:
            src_path = Path(src)
            if src_path.exists():
                dest_path = usb_dir / dest
                if src_path.is_file():
                    shutil.copy2(src_path, dest_path)
                else:
                    shutil.copytree(src_path, dest_path, dirs_exist_ok=True)
        
        # Create installer scripts
        from installer import USBInstaller
        installer = USBInstaller(usb_dir)
        installer.create_installer_bat()
        installer.create_uninstaller_bat()
        if os.name == 'posix':
            installer.create_linux_installer()
        installer.create_readme()
        
        print(f"✓ USB deployment package created in: {usb_dir}")
        print(f"✓ Total size: {sum(f.stat().st_size for f in usb_dir.rglob('*') if f.is_file()) / 1024 / 1024:.1f} MB")
    else:
        print(f"❌ EXE not found: {exe_src}")
        return False
    
    return True

def create_rthook():
    """Create runtime hook for PyInstaller."""
    rthook_content = """# Runtime hook for Picam
import sys
import os

# Add frozen EXE directory to path
if getattr(sys, 'frozen', False):
    # Running as PyInstaller bundle
    sys.path.insert(0, sys._MEIPASS)
    
    # Set environment variables
    os.environ['PICAM_FROZEN'] = '1'
    os.environ['PICAM_VERSION'] = '1.0.0'
    
    # Create required directories
    import pathlib
    reports_dir = pathlib.Path('picam_reports')
    reports_dir.mkdir(exist_ok=True)
"""
    
    rthook_file = Path("rthook.py")
    rthook_file.write_text(rthook_content)
    return rthook_file

def main():
    """Main build process."""
    print("PICAM v1.0 - BUILD SYSTEM")
    print("=" * 60)
    
    # Step 1: Create security config
    password = input("Enter live test password (press Enter for default): ").strip()
    if not password:
        password = None
    
    security_config = create_security_config(password)
    
    # Step 2: Create runtime hook
    rthook_file = create_rthook()
    
    # Step 3: Build EXE
    print("\n" + "=" * 60)
    print("BUILDING EXECUTABLE")
    print("=" * 60)
    
    build_options = input("Build options:\n"
                         "  1. One-file EXE (Recommended)\n"
                         "  2. Directory bundle\n"
                         "Choice [1]: ").strip() or "1"
    
    console_option = input("Console:\n"
                          "  1. With console (for debugging)\n"
                          "  2. Without console (production)\n"
                          "Choice [2]: ").strip() or "2"
    
    success = build_exe(
        onefile=(build_options == "1"),
        console=(console_option == "1"),
        optimize=True
    )
    
    if success:
        print("\n" + "=" * 60)
        print("BUILD COMPLETE")
        print("=" * 60)
        print("\nNEXT STEPS:")
        print("1. Copy entire 'usb_deployment' folder to USB drive")
        print("2. On target machine, run 'install.bat' (Windows) or './install.sh' (Linux)")
        print("3. Launch Picam and enter live test password")
        print("\nSECURITY NOTES:")
        print("- Change default password before deployment")
        print("- Machine binding will prevent copying to other machines")
        print("- Core logic is frozen and cannot be modified")
    else:
        print("\n❌ Build failed. Check errors above.")

if __name__ == "__main__":
    main()