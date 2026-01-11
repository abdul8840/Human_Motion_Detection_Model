#!/usr/bin/env python3
"""
Picam Computer Vision System - Setup and Run Script
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages."""
    print("Installing required packages...")
    requirements = [
        "opencv-python>=4.8.0",
        "mediapipe>=0.10.0",
        "ultralytics>=8.0.0",
        "numpy>=1.24.0"
    ]
    
    for package in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ Installed: {package}")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install: {package}")
            print("You can install manually: pip install", package)

def create_requirements_file():
    """Create requirements.txt file."""
    requirements = """opencv-python>=4.8.0
mediapipe>=0.10.0
ultralytics>=8.0.0
numpy>=1.24.0
"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements)
    print("✓ Created requirements.txt")

def run_demo():
    """Run the Picam demo."""
    print("\n" + "="*50)
    print("Starting Picam Computer Vision System...")
    print("="*50)
    
    # Import and run the main function
    try:
        from picam import VideoProcessor, main
        main()
    except ImportError as e:
        print(f"Error: {e}")
        print("\nMake sure you have saved the VideoProcessor code as 'picam.py'")
        print("or run this script in the same directory as your VideoProcessor class.")

def main_menu():
    """Display main menu."""
    while True:
        print("\n" + "="*50)
        print("PICAM COMPUTER VISION SYSTEM")
        print("="*50)
        print("1. Install requirements")
        print("2. Create requirements.txt")
        print("3. Run webcam demo")
        print("4. Run with video file")
        print("5. Run with custom settings")
        print("6. Exit")
        print("="*50)
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == "1":
            install_requirements()
        elif choice == "2":
            create_requirements_file()
        elif choice == "3":
            run_webcam_demo()
        elif choice == "4":
            run_video_file_demo()
        elif choice == "5":
            run_custom_demo()
        elif choice == "6":
            print("\nGoodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

def run_webcam_demo():
    """Run webcam demo."""
    try:
        from picam import VideoProcessor
        processor = VideoProcessor(source=0)
        processor.run()
    except ImportError:
        print("Please install requirements first (Option 1)")
    except Exception as e:
        print(f"Error: {e}")

def run_video_file_demo():
    """Run video file demo."""
    video_path = input("Enter video file path: ").strip()
    if not os.path.exists(video_path):
        print(f"File not found: {video_path}")
        return
    
    try:
        from picam import VideoProcessor
        processor = VideoProcessor(source=video_path)
        processor.run()
    except ImportError:
        print("Please install requirements first (Option 1)")
    except Exception as e:
        print(f"Error: {e}")

def run_custom_demo():
    """Run demo with custom settings."""
    print("\nCustom Settings:")
    source = input("Enter source (0 for webcam, or file path): ").strip()
    if source.isdigit():
        source = int(source)
    
    model = input("Enter YOLO model [yolov8n.pt]: ").strip() or "yolov8n.pt"
    confidence = input("Enter confidence threshold [0.5]: ").strip()
    confidence = float(confidence) if confidence else 0.5
    
    try:
        from picam import VideoProcessor
        processor = VideoProcessor(source=source, yolo_model=model, confidence_threshold=confidence)
        processor.run()
    except ImportError:
        print("Please install requirements first (Option 1)")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("Welcome to Picam Computer Vision System!")
    print("\nNote: This script will help you set up and run the system.")
    print("Make sure to save the VideoProcessor class as 'picam.py' first.")
    
    # Check if picam.py exists
    if not os.path.exists("picam.py"):
        print("\n⚠  Warning: 'picam.py' not found in current directory.")
        print("Please save the VideoProcessor class code as 'picam.py'")
        create_file = input("Do you want to create it now? (y/n): ").lower()
        if create_file == 'y':
            # You would need to have the VideoProcessor class code here
            print("\nPlease copy the VideoProcessor class code into a file named 'picam.py'")
            input("Press Enter after you've created the file...")
    
    main_menu()