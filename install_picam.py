# install_picam.py
#!/usr/bin/env python3
"""
Picam Installation Script
Helps fix DLL initialization errors
"""

import subprocess
import sys
import os
import platform

def check_system():
    """Check system information."""
    print("="*50)
    print("System Information:")
    print("="*50)
    print(f"Platform: {platform.platform()}")
    print(f"Python: {sys.version}")
    print(f"Architecture: {platform.architecture()[0]}")
    print("="*50)

def install_opencv():
    """Install OpenCV."""
    print("\nInstalling OpenCV...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python"])
        print("✓ OpenCV installed successfully")
    except:
        print("✗ Failed to install OpenCV")
        print("Try: pip install opencv-python")

def install_mediapipe():
    """Install MediaPipe."""
    print("\nInstalling MediaPipe...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "mediapipe"])
        print("✓ MediaPipe installed successfully")
    except:
        print("✗ Failed to install MediaPipe")
        print("Try: pip install mediapipe")

def install_ultralytics_cpu():
    """Install ultralytics with CPU-only PyTorch to avoid DLL issues."""
    print("\nInstalling ultralytics with CPU-only PyTorch...")
    
    # First uninstall existing torch if any
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "torch", "torchvision", "-y"])
    except:
        pass
    
    # Install CPU-only PyTorch
    try:
        print("Installing CPU-only PyTorch...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cpu"])
        print("✓ PyTorch CPU installed successfully")
    except:
        print("Trying alternative installation...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "torch==2.0.0", "torchvision==0.15.0", "--index-url", "https://download.pytorch.org/whl/cpu"])
            print("✓ PyTorch 2.0.0 CPU installed successfully")
        except:
            print("✗ Failed to install PyTorch")
            print("You may need to install it manually from pytorch.org")
    
    # Install ultralytics
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
        print("✓ ultralytics installed successfully")
    except:
        print("✗ Failed to install ultralytics")

def install_simple():
    """Install simple version without PyTorch."""
    print("\nInstalling simple version (no YOLO)...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python", "mediapipe", "numpy"])
        print("✓ Simple version installed successfully")
        print("Note: Person detection (YOLO) will not be available")
    except:
        print("✗ Failed to install simple version")

def create_requirements_file():
    """Create requirements.txt files."""
    
    # Full requirements
    full_req = """opencv-python>=4.8.0
mediapipe>=0.10.0
numpy>=1.24.0
# For YOLO support (CPU only):
# torch==2.0.0
# torchvision==0.15.0
# ultralytics>=8.0.0
"""
    
    # Simple requirements (no YOLO)
    simple_req = """opencv-python>=4.8.0
mediapipe>=0.10.0
numpy>=1.24.0
"""
    
    with open("requirements_full.txt", "w") as f:
        f.write(full_req)
    
    with open("requirements_simple.txt", "w") as f:
        f.write(simple_req)
    
    print("✓ Created requirements files")
    print("  - requirements_full.txt (with YOLO)")
    print("  - requirements_simple.txt (simple version)")

def test_imports():
    """Test if required packages can be imported."""
    print("\n" + "="*50)
    print("Testing imports...")
    print("="*50)
    
    packages = [
        ("cv2", "opencv-python"),
        ("mediapipe", "mediapipe"),
        ("torch", "torch"),
        ("ultralytics", "ultralytics"),
    ]
    
    for module, package in packages:
        try:
            __import__(module)
            print(f"✓ {package} imported successfully")
        except ImportError:
            print(f"✗ {package} not installed")
        except Exception as e:
            print(f"⚠ {package} error: {e}")

def main():
    """Main installation menu."""
    check_system()
    
    while True:
        print("\n" + "="*50)
        print("PICAM INSTALLATION MENU")
        print("="*50)
        print("1. Install full version (with YOLO - may have DLL issues)")
        print("2. Install CPU-only version (recommended for Windows)")
        print("3. Install simple version (no YOLO, no DLL issues)")
        print("4. Test current installation")
        print("5. Create requirements files")
        print("6. Run Picam now")
        print("7. Exit")
        print("="*50)
        
        choice = input("\nEnter your choice (1-7): ").strip()
        
        if choice == "1":
            install_opencv()
            install_mediapipe()
            install_ultralytics_cpu()
        elif choice == "2":
            install_opencv()
            install_mediapipe()
            # Try CPU-only PyTorch
            install_ultralytics_cpu()
        elif choice == "3":
            install_simple()
        elif choice == "4":
            test_imports()
        elif choice == "5":
            create_requirements_file()
        elif choice == "6":
            print("\nRunning Picam...")
            os.system(f"{sys.executable} picam.py")
        elif choice == "7":
            print("\nGoodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    print("Welcome to Picam Installation Helper!")
    print("This script will help fix DLL initialization errors.")
    main()