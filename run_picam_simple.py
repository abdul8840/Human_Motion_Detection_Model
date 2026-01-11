# run_picam_simple.py
"""
Simple runner for Picam without MediaPipe dependency issues.
"""
import cv2
import numpy as np
import time

def run_simple_version():
    """Run Picam with only OpenCV features."""
    print("="*50)
    print("PICAM - Simple Version (No MediaPipe)")
    print("="*50)
    print("Running with OpenCV-only features:")
    print("✓ Motion detection")
    print("✓ Webcam capture")
    print("✓ Basic visualization")
    print("✗ Face detection (requires MediaPipe)")
    print("✗ Person detection (requires YOLO)")
    print("="*50)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return
    
    # Create motion detector
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16)
    
    frame_count = 0
    last_time = time.time()
    fps = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Calculate FPS
            current_time = time.time()
            if frame_count > 0:
                fps = 1 / (current_time - last_time)
            last_time = current_time
            
            # Motion detection
            fg_mask = bg_subtractor.apply(frame)
            _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
            
            # Find motion contours
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw motion
            motion_area = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Min area threshold
                    motion_area += area
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            
            # Calculate motion score
            height, width = frame.shape[:2]
            total_pixels = height * width
            motion_score = motion_area / total_pixels if total_pixels > 0 else 0
            
            # Draw UI
            cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f'Frame: {frame_count}', (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f'Motion: {motion_score*100:.1f}%', (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            if motion_score > 0.05:  # Rapid motion threshold
                cv2.putText(frame, 'RAPID MOTION!', (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow('Picam - Simple Motion Detection', frame)
            frame_count += 1
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
                
    except KeyboardInterrupt:
        print("\n⏹️ Stopped by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n✅ Processed {frame_count} frames")

if __name__ == "__main__":
    run_simple_version()