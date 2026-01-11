# picam.py
import cv2
import numpy as np
import time
import sys
from typing import Optional, Union

class VideoProcessor:
    def __init__(self, source: Union[int, str] = 0, 
                 confidence_threshold: float = 0.5,
                 use_yolo: bool = True,
                 use_mediapipe: bool = True):
        """
        Initialize the VideoProcessor.
        
        Args:
            source: Webcam index (0) or video file path
            confidence_threshold: Minimum confidence for detections
            use_yolo: Enable YOLO detection (may fail if PyTorch has issues)
            use_mediapipe: Enable MediaPipe face detection
        """
        self.source = source
        self.confidence_threshold = confidence_threshold
        self.running = False
        self.frame_count = 0
        self.fps = 0
        self.last_time = 0
        self.use_yolo = use_yolo
        self.use_mediapipe = use_mediapipe
        
        # Try to initialize MediaPipe Face Detection
        self.face_detection = None
        self.mp_drawing = None
        if use_mediapipe:
            try:
                import mediapipe as mp
                self.mp_face_detection = mp.solutions.face_detection
                self.face_detection = self.mp_face_detection.FaceDetection(
                    model_selection=1,
                    min_detection_confidence=confidence_threshold
                )
                self.mp_drawing = mp.solutions.drawing_utils
                print("✓ MediaPipe Face Detection initialized")
            except ImportError:
                print("⚠ MediaPipe not installed. Face detection disabled.")
                print("Install with: pip install mediapipe")
                self.use_mediapipe = False
            except Exception as e:
                print(f"⚠ Error initializing MediaPipe: {e}")
                self.use_mediapipe = False
        
        # Try to initialize YOLOv8 model
        self.yolo_model = None
        if use_yolo:
            try:
                from ultralytics import YOLO
                # Try to load a smaller model first
                try:
                    self.yolo_model = YOLO('yolov8n.pt')
                    print("✓ YOLOv8 model initialized")
                except:
                    # If model file doesn't exist, download it
                    print("Downloading YOLOv8 model...")
                    self.yolo_model = YOLO('yolov8n.pt')
                    print("✓ YOLOv8 model downloaded and initialized")
            except ImportError:
                print("⚠ ultralytics not installed. YOLO detection disabled.")
                print("Install with: pip install ultralytics")
                self.use_yolo = False
            except Exception as e:
                print(f"⚠ Error initializing YOLOv8: {e}")
                print("You can try: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
                self.use_yolo = False
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video source: {source}")
        
        # Get video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps_video = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
        
        print(f"✓ Video source: {source}")
        print(f"✓ Resolution: {self.width}x{self.height}")
        print(f"✓ Face Detection: {'Enabled' if self.use_mediapipe else 'Disabled'}")
        print(f"✓ Person Detection: {'Enabled' if self.use_yolo else 'Disabled'}")
    
    def draw_face_detection(self, frame: np.ndarray, results) -> np.ndarray:
        """Draw MediaPipe face detection bounding boxes."""
        if results and results.detections:
            for detection in results.detections:
                # Get bounding box
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y = int(bboxC.xmin * iw), int(bboxC.ymin * ih)
                w, h = int(bboxC.width * iw), int(bboxC.height * ih)
                
                # Ensure coordinates are within frame boundaries
                x, y = max(0, x), max(0, y)
                w, h = min(iw - x, w), min(ih - y, h)
                
                # Draw rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw confidence
                conf = detection.score[0]
                cv2.putText(frame, f'Face: {conf:.2f}', 
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 255, 0), 2)
        return frame
    
    def draw_yolo_detections(self, frame: np.ndarray, results) -> np.ndarray:
        """Draw YOLOv8 person detections."""
        if results:
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Filter for 'person' class (class 0 in COCO)
                        if int(box.cls) == 0 and box.conf > self.confidence_threshold:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = box.conf[0]
                            
                            # Ensure coordinates are within frame boundaries
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(self.width, x2), min(self.height, y2)
                            
                            # Draw rectangle
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            
                            # Draw label
                            label = f'Person: {conf:.2f}'
                            cv2.putText(frame, label, 
                                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                       0.5, (255, 0, 0), 2)
        return frame
    
    def calculate_fps(self):
        """Calculate and display FPS."""
        current_time = time.time()
        if self.last_time > 0:
            self.fps = 1 / (current_time - self.last_time)
        self.last_time = current_time
    
    def draw_ui(self, frame: np.ndarray, face_count: int, person_count: int) -> np.ndarray:
        """Draw UI elements on frame."""
        # Draw FPS
        fps_text = f'FPS: {self.fps:.1f}'
        cv2.putText(frame, fps_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw statistics
        stats_text = f'Faces: {face_count} | Persons: {person_count}'
        cv2.putText(frame, stats_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw frame count
        frame_text = f'Frame: {self.frame_count}'
        cv2.putText(frame, frame_text, (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        # Draw status indicators
        y_offset = 120
        if self.use_mediapipe:
            cv2.putText(frame, 'Face Detection: ON', (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25
        
        if self.use_yolo:
            cv2.putText(frame, 'Person Detection: ON', (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        return frame
    
    def count_detections(self, face_results, yolo_results) -> tuple:
        """Count the number of faces and persons detected."""
        face_count = 0
        person_count = 0
        
        if face_results and face_results.detections:
            face_count = len(face_results.detections)
        
        if yolo_results:
            for result in yolo_results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        if int(box.cls) == 0 and box.conf > self.confidence_threshold:
                            person_count += 1
        
        return face_count, person_count
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame with face and person detection."""
        face_results = None
        yolo_results = None
        
        # Convert BGR to RGB for MediaPipe if needed
        if self.use_mediapipe and self.face_detection:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_results = self.face_detection.process(rgb_frame)
        
        # Process with YOLOv8 if available
        if self.use_yolo and self.yolo_model:
            try:
                yolo_results = self.yolo_model(frame, verbose=False)
            except Exception as e:
                print(f"YOLO processing error: {e}")
                self.use_yolo = False
        
        # Count detections
        face_count, person_count = self.count_detections(face_results, yolo_results)
        
        # Draw detections
        if self.use_mediapipe:
            frame = self.draw_face_detection(frame, face_results)
        
        if self.use_yolo:
            frame = self.draw_yolo_detections(frame, yolo_results)
        
        # Calculate and draw UI
        self.calculate_fps()
        frame = self.draw_ui(frame, face_count, person_count)
        
        self.frame_count += 1
        return frame
    
    def run(self):
        """Main processing loop."""
        self.running = True
        print("\n" + "="*50)
        print("PICAM - Computer Vision System")
        print("="*50)
        print("Controls:")
        print("  'q' or ESC - Quit")
        print("  's' - Save screenshot")
        print("  'p' - Pause/resume")
        print("  'f' - Toggle fullscreen")
        print("  '1' - Toggle face detection")
        print("  '2' - Toggle person detection")
        print("="*50 + "\n")
        
        paused = False
        fullscreen = False
        window_name = 'Picam - Computer Vision System'
        
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        try:
            while self.running:
                if not paused:
                    ret, frame = self.cap.read()
                    if not ret:
                        if isinstance(self.source, str):
                            print("End of video file reached.")
                            # Option to loop the video
                            replay = input("Replay video? (y/n): ").lower()
                            if replay == 'y':
                                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                                continue
                        break
                    
                    # Process frame
                    processed_frame = self.process_frame(frame)
                    
                    # Display frame
                    cv2.imshow(window_name, processed_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                elif key == ord('s'):
                    # Save screenshot
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"picam_screenshot_{timestamp}.jpg"
                    cv2.imwrite(filename, processed_frame)
                    print(f"✓ Screenshot saved: {filename}")
                elif key == ord('p'):
                    paused = not paused
                    status = "⏸️ Paused" if paused else "▶️ Resumed"
                    print(f"{status}")
                elif key == ord('f'):
                    fullscreen = not fullscreen
                    if fullscreen:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    else:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    print(f"Fullscreen: {'ON' if fullscreen else 'OFF'}")
                elif key == ord('1'):
                    self.use_mediapipe = not self.use_mediapipe
                    print(f"Face detection: {'ON' if self.use_mediapipe else 'OFF'}")
                elif key == ord('2'):
                    self.use_yolo = not self.use_yolo
                    print(f"Person detection: {'ON' if self.use_yolo else 'OFF'}")
                
        except KeyboardInterrupt:
            print("\n⏹️ Interrupted by user")
        except Exception as e:
            print(f"\n❌ Error during processing: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Clean up resources."""
        self.running = False
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        if self.face_detection:
            self.face_detection.close()
        print(f"\n✅ VideoProcessor stopped")
        print(f"📊 Total frames processed: {self.frame_count}")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.stop()


def simple_main():
    """Simple main function without complex dependencies."""
    print("Picam Computer Vision System - Simple Mode")
    print("\nOptions:")
    print("1. Run with webcam (simple OpenCV only)")
    print("2. Run with webcam (try with AI detection)")
    print("3. Run with video file")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        # Simple webcam without AI
        run_simple_webcam()
    elif choice == "2":
        # Try with AI detection
        try:
            processor = VideoProcessor(source=0)
            processor.run()
        except Exception as e:
            print(f"Error: {e}")
            print("\nFalling back to simple webcam...")
            run_simple_webcam()
    elif choice == "3":
        video_path = input("Enter video file path: ").strip()
        try:
            processor = VideoProcessor(source=video_path)
            processor.run()
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Invalid choice")


def run_simple_webcam():
    """Simple webcam feed without AI detection."""
    print("\nStarting simple webcam feed...")
    print("Press 'q' to quit")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return
    
    frame_count = 0
    last_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - last_time) if frame_count > 0 else 0
            last_time = current_time
            
            # Draw FPS
            cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f'Frame: {frame_count}', (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, 'Simple Mode - Press q to quit', (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)
            
            cv2.imshow('Picam - Simple Webcam', frame)
            frame_count += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"Total frames: {frame_count}")


if __name__ == "__main__":
    print("="*50)
    print("PICAM COMPUTER VISION SYSTEM")
    print("="*50)
    
    try:
        # Try to run the full version
        processor = VideoProcessor(source=0)
        processor.run()
    except ImportError as e:
        print(f"\n⚠ Missing package: {e}")
        print("\nFor full functionality, install:")
        print("pip install opencv-python mediapipe ultralytics")
        print("\nFor now, running in simple mode...")
        run_simple_webcam()
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure webcam is connected")
        print("2. Try a different camera index (0, 1, 2)")
        print("3. Check if camera is in use by another app")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("\nRunning in simple mode...")
        run_simple_webcam()