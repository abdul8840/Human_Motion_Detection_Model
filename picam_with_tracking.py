# picam_with_tracking.py
import cv2
import numpy as np
import time
import sys
from typing import Optional, Union, List, Tuple, Dict
from collections import OrderedDict
import math

class EuclideanTracker:
    """Simple Euclidean distance-based tracker for objects."""
    def __init__(self, max_disappeared: int = 10, max_distance: float = 50.0):
        """
        Initialize the tracker.
        
        Args:
            max_disappeared: Maximum frames an object can disappear before removing
            max_distance: Maximum Euclidean distance to match objects between frames
        """
        self.next_object_id = 0
        self.objects = OrderedDict()  # id -> (centroid, bbox, confidence, class_type, disappeared_frames)
        self.disappeared = OrderedDict()  # id -> disappeared frames count
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.colors = {}  # id -> color for visualization
        
    def _get_color(self, object_id: int) -> Tuple[int, int, int]:
        """Get a consistent color for an object ID."""
        if object_id not in self.colors:
            # Generate a color based on object ID
            color_idx = object_id * 30 % 180  # Use hue for distinct colors
            self.colors[object_id] = self.hsv_to_rgb(color_idx, 255, 255)
        return self.colors[object_id]
    
    def hsv_to_rgb(self, h: int, s: int, v: int) -> Tuple[int, int, int]:
        """Convert HSV to RGB color."""
        h = h % 180  # OpenCV uses 0-180 for hue
        hsv_color = np.uint8([[[h, s, v]]])
        rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)
        return (int(rgb_color[0][0][0]), int(rgb_color[0][0][1]), int(rgb_color[0][0][2]))
    
    def register(self, centroid: Tuple[float, float], bbox: Tuple[int, int, int, int], 
                 confidence: float, class_type: str = "person") -> int:
        """Register a new object."""
        object_id = self.next_object_id
        self.next_object_id += 1
        
        self.objects[object_id] = {
            'centroid': centroid,
            'bbox': bbox,
            'confidence': confidence,
            'class_type': class_type,
            'last_seen': time.time(),
            'first_seen': time.time(),
            'track_count': 0
        }
        self.disappeared[object_id] = 0
        
        return object_id
    
    def deregister(self, object_id: int):
        """Remove an object from tracking."""
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.disappeared:
            del self.disappeared[object_id]
        if object_id in self.colors:
            del self.colors[object_id]
    
    def update(self, detections: List[Dict]) -> Dict[int, Dict]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of dicts with keys: 'centroid', 'bbox', 'confidence', 'class_type'
        
        Returns:
            Dictionary of tracked objects with their IDs
        """
        # If no detections, mark all objects as disappeared
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects.copy()
        
        # If no objects currently tracked, register all detections
        if len(self.objects) == 0:
            for detection in detections:
                self.register(detection['centroid'], detection['bbox'], 
                            detection['confidence'], detection.get('class_type', 'person'))
            return self.objects.copy()
        
        # Match existing objects with new detections using Euclidean distance
        object_ids = list(self.objects.keys())
        object_centroids = [self.objects[oid]['centroid'] for oid in object_ids]
        detection_centroids = [d['centroid'] for d in detections]
        
        # Compute pairwise distances
        distances = np.zeros((len(object_ids), len(detections)))
        for i, obj_centroid in enumerate(object_centroids):
            for j, det_centroid in enumerate(detection_centroids):
                distances[i, j] = math.sqrt(
                    (obj_centroid[0] - det_centroid[0])**2 + 
                    (obj_centroid[1] - det_centroid[1])**2
                )
        
        # Match objects to detections (simple greedy matching)
        used_detections = set()
        used_objects = set()
        
        # Sort by distance for matching
        matches = []
        while True:
            if distances.size == 0:
                break
            
            # Find minimum distance
            min_idx = np.unravel_index(np.argmin(distances), distances.shape)
            min_distance = distances[min_idx]
            
            if min_distance > self.max_distance:
                break
            
            object_idx, detection_idx = min_idx
            object_id = object_ids[object_idx]
            
            # Match if both are unmatched
            if object_idx not in used_objects and detection_idx not in used_detections:
                matches.append((object_id, detection_idx, min_distance))
                used_objects.add(object_idx)
                used_detections.add(detection_idx)
            
            # Set matched rows and columns to infinity
            distances[object_idx, :] = float('inf')
            distances[:, detection_idx] = float('inf')
        
        # Update matched objects
        for object_id, detection_idx, distance in matches:
            detection = detections[detection_idx]
            
            # Update object properties
            self.objects[object_id]['centroid'] = detection['centroid']
            self.objects[object_id]['bbox'] = detection['bbox']
            self.objects[object_id]['confidence'] = detection['confidence']
            self.objects[object_id]['last_seen'] = time.time()
            self.objects[object_id]['track_count'] += 1
            
            # Reset disappeared counter
            self.disappeared[object_id] = 0
        
        # Register unmatched detections as new objects
        for j, detection in enumerate(detections):
            if j not in used_detections:
                self.register(detection['centroid'], detection['bbox'], 
                            detection['confidence'], detection.get('class_type', 'person'))
        
        # Mark unmatched objects as disappeared
        for i, object_id in enumerate(object_ids):
            if i not in used_objects:
                self.disappeared[object_id] += 1
                # Predict next position (simple linear prediction)
                if self.disappeared[object_id] < 3:  # Only predict for recently disappeared
                    # Store predicted position (for visualization)
                    self.objects[object_id]['predicted'] = True
                else:
                    self.objects[object_id]['predicted'] = False
                
                # Remove if disappeared too long
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
        
        return self.objects.copy()
    
    def get_stats(self, object_id: int) -> Dict:
        """Get statistics for an object."""
        if object_id in self.objects:
            obj = self.objects[object_id]
            return {
                'track_duration': time.time() - obj['first_seen'],
                'track_count': obj['track_count'],
                'disappeared': self.disappeared.get(object_id, 0)
            }
        return {}


class VideoProcessor:
    def __init__(self, source: Union[int, str] = 0, 
                 confidence_threshold: float = 0.5,
                 use_yolo: bool = True,
                 use_mediapipe: bool = True,
                 enable_tracking: bool = True,
                 tracker_max_distance: float = 50.0,
                 tracker_max_disappeared: int = 15):
        """
        Initialize the VideoProcessor with tracking.
        
        Args:
            source: Webcam index (0) or video file path
            confidence_threshold: Minimum confidence for detections
            use_yolo: Enable YOLO detection
            use_mediapipe: Enable MediaPipe face detection
            enable_tracking: Enable object tracking
            tracker_max_distance: Max distance for object matching
            tracker_max_disappeared: Max frames before removing disappeared object
        """
        self.source = source
        self.confidence_threshold = confidence_threshold
        self.running = False
        self.frame_count = 0
        self.fps = 0
        self.last_time = 0
        self.use_yolo = use_yolo
        self.use_mediapipe = use_mediapipe
        self.enable_tracking = enable_tracking
        
        # Initialize trackers
        if enable_tracking:
            self.face_tracker = EuclideanTracker(
                max_disappeared=tracker_max_disappeared,
                max_distance=tracker_max_distance
            )
            self.person_tracker = EuclideanTracker(
                max_disappeared=tracker_max_disappeared,
                max_distance=tracker_max_distance
            )
            print("✓ Object tracking initialized")
        
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
                try:
                    self.yolo_model = YOLO('yolov8n.pt')
                    print("✓ YOLOv8 model initialized")
                except:
                    print("Downloading YOLOv8 model...")
                    self.yolo_model = YOLO('yolov8n.pt')
                    print("✓ YOLOv8 model downloaded and initialized")
            except ImportError:
                print("⚠ ultralytics not installed. YOLO detection disabled.")
                print("Install with: pip install ultralytics")
                self.use_yolo = False
            except Exception as e:
                print(f"⚠ Error initializing YOLOv8: {e}")
                self.use_yolo = False
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video source: {source}")
        
        # Get video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps_video = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
        
        # Statistics
        self.total_faces_tracked = 0
        self.total_persons_tracked = 0
        
        print(f"✓ Video source: {source}")
        print(f"✓ Resolution: {self.width}x{self.height}")
        print(f"✓ Face Detection: {'Enabled' if self.use_mediapipe else 'Disabled'}")
        print(f"✓ Person Detection: {'Enabled' if self.use_yolo else 'Disabled'}")
        print(f"✓ Tracking: {'Enabled' if enable_tracking else 'Disabled'}")
    
    def get_centroid(self, bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """Calculate centroid from bounding box (x1, y1, x2, y2)."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def extract_face_detections(self, frame: np.ndarray, results) -> List[Dict]:
        """Extract face detections for tracking."""
        detections = []
        if results and results.detections:
            for detection in results.detections:
                # Get bounding box
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x = int(bboxC.xmin * iw)
                y = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)
                
                # Ensure coordinates are within frame boundaries
                x, y = max(0, x), max(0, y)
                w, h = min(iw - x, w), min(ih - y, h)
                
                bbox = (x, y, x + w, y + h)
                centroid = self.get_centroid(bbox)
                confidence = float(detection.score[0])
                
                if confidence >= self.confidence_threshold:
                    detections.append({
                        'centroid': centroid,
                        'bbox': bbox,
                        'confidence': confidence,
                        'class_type': 'face'
                    })
        return detections
    
    def extract_person_detections(self, frame: np.ndarray, results) -> List[Dict]:
        """Extract person detections for tracking."""
        detections = []
        if results:
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Filter for 'person' class (class 0 in COCO)
                        if int(box.cls) == 0 and box.conf > self.confidence_threshold:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = float(box.conf[0])
                            
                            # Ensure coordinates are within frame boundaries
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(self.width, x2), min(self.height, y2)
                            
                            bbox = (x1, y1, x2, y2)
                            centroid = self.get_centroid(bbox)
                            
                            detections.append({
                                'centroid': centroid,
                                'bbox': bbox,
                                'confidence': conf,
                                'class_type': 'person'
                            })
        return detections
    
    def draw_tracked_object(self, frame: np.ndarray, object_id: int, 
                          obj_info: Dict, class_type: str) -> np.ndarray:
        """Draw a tracked object with its ID and information."""
        color = (0, 255, 0) if class_type == 'face' else (255, 0, 0)
        
        # Get tracker color if enabled
        if self.enable_tracking:
            if class_type == 'face':
                color = self.face_tracker._get_color(object_id)
            else:
                color = self.person_tracker._get_color(object_id)
        
        # Draw bounding box
        x1, y1, x2, y2 = obj_info['bbox']
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw ID and confidence
        conf = obj_info['confidence']
        id_text = f"ID: {object_id}"
        conf_text = f"{conf:.2f}"
        
        # Draw ID box
        (text_width, text_height), baseline = cv2.getTextSize(
            id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )
        
        # Background for ID
        cv2.rectangle(frame, 
                     (x1, y1 - text_height - 10),
                     (x1 + text_width + 10, y1),
                     color, -1)
        
        # Draw ID text
        cv2.putText(frame, id_text, 
                   (x1 + 5, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw confidence below ID
        cv2.putText(frame, conf_text, 
                   (x1, y1 - text_height - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw centroid point
        centroid = obj_info['centroid']
        cv2.circle(frame, (int(centroid[0]), int(centroid[1])), 3, color, -1)
        
        # Draw trail if tracking is enabled
        if self.enable_tracking and 'trail' in obj_info:
            trail = obj_info['trail']
            if len(trail) > 1:
                for i in range(1, len(trail)):
                    cv2.line(frame, trail[i-1], trail[i], color, 1)
        
        return frame
    
    def update_trail(self, object_id: int, centroid: Tuple[float, float], 
                    class_type: str, max_trail_length: int = 20):
        """Update movement trail for an object."""
        if not self.enable_tracking:
            return
        
        tracker = self.face_tracker if class_type == 'face' else self.person_tracker
        
        if object_id in tracker.objects:
            centroid_point = (int(centroid[0]), int(centroid[1]))
            
            if 'trail' not in tracker.objects[object_id]:
                tracker.objects[object_id]['trail'] = []
            
            tracker.objects[object_id]['trail'].append(centroid_point)
            
            # Limit trail length
            if len(tracker.objects[object_id]['trail']) > max_trail_length:
                tracker.objects[object_id]['trail'].pop(0)
    
    def calculate_fps(self):
        """Calculate and display FPS."""
        current_time = time.time()
        if self.last_time > 0:
            self.fps = 1 / (current_time - self.last_time)
        self.last_time = current_time
    
    def draw_ui(self, frame: np.ndarray, tracked_faces: Dict, tracked_persons: Dict) -> np.ndarray:
        """Draw UI elements on frame."""
        # Draw FPS
        fps_text = f'FPS: {self.fps:.1f}'
        cv2.putText(frame, fps_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw statistics
        active_faces = len([oid for oid in tracked_faces 
                          if self.face_tracker.disappeared.get(oid, 0) == 0]) if self.enable_tracking else 0
        active_persons = len([oid for oid in tracked_persons 
                            if self.person_tracker.disappeared.get(oid, 0) == 0]) if self.enable_tracking else 0
        
        stats_text = f'Faces: {active_faces} | Persons: {active_persons}'
        cv2.putText(frame, stats_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw tracking info
        if self.enable_tracking:
            track_text = f'Tracked Faces: {len(tracked_faces)} | Tracked Persons: {len(tracked_persons)}'
            cv2.putText(frame, track_text, (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 100), 2)
            
            # Draw frame count
            frame_text = f'Frame: {self.frame_count}'
            cv2.putText(frame, frame_text, (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            
            # Draw legend
            legend_y = self.height - 100
            cv2.putText(frame, 'Legend:', (10, legend_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, 'Green - Face with ID', (10, legend_y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, 'Blue - Person with ID', (10, legend_y + 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(frame, 'Dot - Centroid', (10, legend_y + 65), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame with detection and tracking."""
        face_detections = []
        person_detections = []
        
        # Convert BGR to RGB for MediaPipe if needed
        if self.use_mediapipe and self.face_detection:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_results = self.face_detection.process(rgb_frame)
            face_detections = self.extract_face_detections(frame, face_results)
        
        # Process with YOLOv8 if available
        if self.use_yolo and self.yolo_model:
            try:
                yolo_results = self.yolo_model(frame, verbose=False)
                person_detections = self.extract_person_detections(frame, yolo_results)
            except Exception as e:
                print(f"YOLO processing error: {e}")
                self.use_yolo = False
        
        # Update trackers if enabled
        tracked_faces = {}
        tracked_persons = {}
        
        if self.enable_tracking:
            # Update face tracker
            if self.use_mediapipe:
                tracked_faces = self.face_tracker.update(face_detections)
                # Draw tracked faces
                for obj_id, obj_info in tracked_faces.items():
                    if self.face_tracker.disappeared.get(obj_id, 0) == 0:
                        frame = self.draw_tracked_object(frame, obj_id, obj_info, 'face')
                        # Update trail
                        self.update_trail(obj_id, obj_info['centroid'], 'face')
                    else:
                        # Draw faded for disappeared objects
                        faded_color = tuple(c // 3 for c in self.face_tracker._get_color(obj_id))
                        x1, y1, x2, y2 = obj_info['bbox']
                        cv2.rectangle(frame, (x1, y1), (x2, y2), faded_color, 1)
                        cv2.putText(frame, f"ID: {obj_id} (lost)", 
                                  (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.5, faded_color, 1)
            
            # Update person tracker
            if self.use_yolo:
                tracked_persons = self.person_tracker.update(person_detections)
                # Draw tracked persons
                for obj_id, obj_info in tracked_persons.items():
                    if self.person_tracker.disappeared.get(obj_id, 0) == 0:
                        frame = self.draw_tracked_object(frame, obj_id, obj_info, 'person')
                        # Update trail
                        self.update_trail(obj_id, obj_info['centroid'], 'person')
                    else:
                        # Draw faded for disappeared objects
                        faded_color = tuple(c // 3 for c in self.person_tracker._get_color(obj_id))
                        x1, y1, x2, y2 = obj_info['bbox']
                        cv2.rectangle(frame, (x1, y1), (x2, y2), faded_color, 1)
                        cv2.putText(frame, f"ID: {obj_id} (lost)", 
                                  (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.5, faded_color, 1)
        else:
            # Draw detections without tracking
            if self.use_mediapipe and face_results:
                for detection in face_detections:
                    x1, y1, x2, y2 = detection['bbox']
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'Face: {detection["confidence"]:.2f}', 
                               (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.5, (0, 255, 0), 2)
            
            if self.use_yolo:
                for detection in person_detections:
                    x1, y1, x2, y2 = detection['bbox']
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f'Person: {detection["confidence"]:.2f}', 
                               (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.5, (255, 0, 0), 2)
        
        # Calculate and draw UI
        self.calculate_fps()
        frame = self.draw_ui(frame, tracked_faces, tracked_persons)
        
        self.frame_count += 1
        return frame
    
    def print_tracking_stats(self):
        """Print tracking statistics."""
        if self.enable_tracking:
            print("\n" + "="*50)
            print("TRACKING STATISTICS")
            print("="*50)
            
            if self.use_mediapipe:
                print(f"\nFace Tracking:")
                print(f"  Active faces: {len([oid for oid in self.face_tracker.objects 
                                            if self.face_tracker.disappeared.get(oid, 0) == 0])}")
                print(f"  Total faces tracked: {self.face_tracker.next_object_id}")
                print(f"  Currently lost: {len([oid for oid in self.face_tracker.objects 
                                              if self.face_tracker.disappeared.get(oid, 0) > 0])}")
            
            if self.use_yolo:
                print(f"\nPerson Tracking:")
                print(f"  Active persons: {len([oid for oid in self.person_tracker.objects 
                                              if self.person_tracker.disappeared.get(oid, 0) == 0])}")
                print(f"  Total persons tracked: {self.person_tracker.next_object_id}")
                print(f"  Currently lost: {len([oid for oid in self.person_tracker.objects 
                                              if self.person_tracker.disappeared.get(oid, 0) > 0])}")
            
            print("="*50)
    
    def run(self):
        """Main processing loop."""
        self.running = True
        print("\n" + "="*50)
        print("PICAM - Computer Vision System with Tracking")
        print("="*50)
        print("Controls:")
        print("  'q' or ESC - Quit")
        print("  's' - Save screenshot")
        print("  'p' - Pause/resume")
        print("  't' - Toggle tracking")
        print("  'd' - Display tracking statistics")
        print("  'c' - Clear all tracks")
        print("  '1' - Toggle face detection")
        print("  '2' - Toggle person detection")
        print("="*50 + "\n")
        
        paused = False
        window_name = 'Picam - Object Tracking'
        
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        try:
            while self.running:
                if not paused:
                    ret, frame = self.cap.read()
                    if not ret:
                        if isinstance(self.source, str):
                            print("End of video file reached.")
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
                    filename = f"picam_tracking_{timestamp}.jpg"
                    cv2.imwrite(filename, processed_frame)
                    print(f"✓ Screenshot saved: {filename}")
                elif key == ord('p'):
                    paused = not paused
                    status = "⏸️ Paused" if paused else "▶️ Resumed"
                    print(f"{status}")
                elif key == ord('t'):
                    self.enable_tracking = not self.enable_tracking
                    status = "ENABLED" if self.enable_tracking else "DISABLED"
                    print(f"Tracking {status}")
                elif key == ord('d'):
                    self.print_tracking_stats()
                elif key == ord('c'):
                    if self.enable_tracking:
                        if self.use_mediapipe:
                            self.face_tracker = EuclideanTracker(
                                max_disappeared=self.face_tracker.max_disappeared,
                                max_distance=self.face_tracker.max_distance
                            )
                        if self.use_yolo:
                            self.person_tracker = EuclideanTracker(
                                max_disappeared=self.person_tracker.max_disappeared,
                                max_distance=self.person_tracker.max_distance
                            )
                        print("✓ All tracks cleared")
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
        
        # Print final statistics
        print(f"\n✅ VideoProcessor stopped")
        print(f"📊 Total frames processed: {self.frame_count}")
        
        if self.enable_tracking:
            self.print_tracking_stats()
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.stop()


def main():
    """Main function to run the VideoProcessor with tracking."""
    print("="*50)
    print("PICAM - Object Tracking System")
    print("="*50)
    
    try:
        # Create processor with tracking enabled
        processor = VideoProcessor(
            source=0,  # Webcam
            confidence_threshold=0.5,
            use_yolo=True,
            use_mediapipe=True,
            enable_tracking=True,
            tracker_max_distance=50.0,
            tracker_max_disappeared=15
        )
        
        # Run the processor
        processor.run()
    except ImportError as e:
        print(f"\n⚠ Missing package: {e}")
        print("\nInstall required packages:")
        print("pip install opencv-python mediapipe ultralytics numpy")
    except ValueError as e:
        print(f"\n❌ Error: {e}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()