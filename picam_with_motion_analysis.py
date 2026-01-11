# picam_motion_only.py
import cv2
import numpy as np
import time
import sys
from typing import Optional, Union, List, Tuple, Dict, Any
from collections import OrderedDict
import math

class MotionDetector:
    """Motion detection using background subtraction."""
    def __init__(self, history: int = 500, var_threshold: int = 16, 
                 detect_shadows: bool = True, high_motion_threshold: float = 0.05,
                 min_motion_area: int = 100):
        """
        Initialize motion detector.
        """
        # Background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows
        )
        
        # Motion parameters
        self.high_motion_threshold = high_motion_threshold
        self.min_motion_area = min_motion_area
        
        # Motion history
        self.motion_history = []
        self.max_history = 30
        
        # Frame statistics
        self.frame_width = 0
        self.frame_height = 0
        self.total_pixels = 0
        
        # Debug visualization
        self.visualize = True
        
    def set_frame_size(self, width: int, height: int):
        """Set frame dimensions."""
        self.frame_width = width
        self.frame_height = height
        self.total_pixels = width * height
    
    def calculate_motion_score(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Calculate motion score for the current frame.
        """
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Remove shadows (value 127)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours of motion areas
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        motion_areas = []
        motion_centroids = []
        valid_contours = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_motion_area:
                motion_areas.append(area)
                valid_contours.append(contour)
                
                # Calculate centroid
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    motion_centroids.append((cx, cy))
        
        # Calculate total motion area
        total_motion_area = sum(motion_areas)
        
        # Calculate motion score (percentage of frame with motion)
        if self.total_pixels > 0:
            motion_score = total_motion_area / self.total_pixels
        else:
            motion_score = 0.0
        
        # Check for rapid motion
        rapid_motion = motion_score > self.high_motion_threshold
        
        # Update motion history
        self.motion_history.append(motion_score)
        if len(self.motion_history) > self.max_history:
            self.motion_history.pop(0)
        
        # Calculate average motion
        avg_motion = np.mean(self.motion_history) if self.motion_history else 0
        
        return {
            'motion_score': float(motion_score),
            'motion_percentage': float(motion_score * 100),
            'motion_areas': motion_areas,
            'motion_centroids': motion_centroids,
            'rapid_motion': rapid_motion,
            'avg_motion': float(avg_motion),
            'motion_mask': fg_mask,
            'contours': valid_contours,
            'total_motion_area': int(total_motion_area),
            'num_motion_regions': len(valid_contours)
        }
    
    def check_motion_near_faces(self, motion_data: Dict, face_bboxes: List[Tuple]) -> Dict[str, Any]:
        """
        Check if motion occurs near faces.
        """
        if not motion_data['contours'] or not face_bboxes:
            return {
                'motion_near_faces': False,
                'affected_faces': [],
                'motion_face_distances': [],
                'closest_distance': float('inf')
            }
        
        affected_faces = []
        motion_face_distances = []
        closest_distance = float('inf')
        
        # Define expansion radius around face (percentage of face size)
        for face_idx, face_bbox in enumerate(face_bboxes):
            x1, y1, x2, y2 = face_bbox
            face_width = x2 - x1
            face_height = y2 - y1
            
            # Calculate face center
            face_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            # Define expanded region around face
            expansion_factor = 1.5
            expanded_x1 = max(0, x1 - int(face_width * (expansion_factor - 1) / 2))
            expanded_y1 = max(0, y1 - int(face_height * (expansion_factor - 1) / 2))
            expanded_x2 = min(self.frame_width, x2 + int(face_width * (expansion_factor - 1) / 2))
            expanded_y2 = min(self.frame_height, y2 + int(face_height * (expansion_factor - 1) / 2))
            
            # Check each motion contour
            motion_near_this_face = False
            min_distance_to_face = float('inf')
            
            for contour in motion_data['contours']:
                # Get bounding rect of contour
                x, y, w, h = cv2.boundingRect(contour)
                contour_center = (x + w // 2, y + h // 2)
                
                # Calculate distance from contour center to face center
                distance = math.sqrt(
                    (contour_center[0] - face_center[0])**2 + 
                    (contour_center[1] - face_center[1])**2
                )
                
                # Check if contour overlaps with expanded face region
                contour_in_region = (
                    x < expanded_x2 and x + w > expanded_x1 and
                    y < expanded_y2 and y + h > expanded_y1
                )
                
                if contour_in_region:
                    motion_near_this_face = True
                    min_distance_to_face = min(min_distance_to_face, distance)
            
            if motion_near_this_face:
                affected_faces.append(face_idx)
                motion_face_distances.append(min_distance_to_face)
                closest_distance = min(closest_distance, min_distance_to_face)
        
        return {
            'motion_near_faces': len(affected_faces) > 0,
            'affected_faces': affected_faces,
            'motion_face_distances': motion_face_distances,
            'closest_distance': closest_distance if closest_distance != float('inf') else 0
        }
    
    def visualize_motion(self, frame: np.ndarray, motion_data: Dict, 
                        motion_near_faces_data: Dict, face_bboxes: List[Tuple]) -> np.ndarray:
        """Visualize motion detection results on frame."""
        if not self.visualize:
            return frame
        
        # Create visualization overlay
        overlay = frame.copy()
        
        # Draw motion contours
        if motion_data['contours']:
            cv2.drawContours(overlay, motion_data['contours'], -1, (0, 255, 255), 2)
            
            # Draw motion centroids
            for centroid in motion_data['motion_centroids']:
                cv2.circle(overlay, centroid, 5, (0, 255, 255), -1)
        
        # Draw expanded face regions where motion is detected
        for face_idx in motion_near_faces_data['affected_faces']:
            if face_idx < len(face_bboxes):
                x1, y1, x2, y2 = face_bboxes[face_idx]
                face_width = x2 - x1
                face_height = y2 - y1
                
                # Expanded region
                expansion_factor = 1.5
                expanded_x1 = max(0, x1 - int(face_width * (expansion_factor - 1) / 2))
                expanded_y1 = max(0, y1 - int(face_height * (expansion_factor - 1) / 2))
                expanded_x2 = min(self.frame_width, x2 + int(face_width * (expansion_factor - 1) / 2))
                expanded_y2 = min(self.frame_height, y2 + int(face_height * (expansion_factor - 1) / 2))
                
                # Draw expanded region
                cv2.rectangle(overlay, (expanded_x1, expanded_y1), (expanded_x2, expanded_y2), 
                            (255, 0, 255), 2)
                
                # Label with distance
                if face_idx < len(motion_near_faces_data['motion_face_distances']):
                    distance = motion_near_faces_data['motion_face_distances'][face_idx]
                    cv2.putText(overlay, f'Dist: {distance:.1f}', 
                              (expanded_x1, expanded_y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        # Apply overlay with transparency
        alpha = 0.3
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        return frame


class FaceDetector:
    """Face detection using OpenCV Haar cascades."""
    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        
        # Load face cascade
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            if self.face_cascade.empty():
                raise Exception("Could not load face cascade")
            print("✓ OpenCV Face Detection initialized")
            self.available = True
        except Exception as e:
            print(f"⚠ OpenCV face detection error: {e}")
            self.available = False
    
    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """Detect faces in frame."""
        if not self.available:
            return []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        detections = []
        for (x, y, w, h) in faces:
            bbox = (x, y, x + w, y + h)
            centroid = ((x + x + w) / 2, (y + y + h) / 2)
            detections.append({
                'centroid': centroid,
                'bbox': bbox,
                'confidence': 0.9,  # OpenCV doesn't provide confidence
                'class_type': 'face'
            })
        
        return detections


class EuclideanTracker:
    """Simple Euclidean distance-based tracker for objects."""
    def __init__(self, max_disappeared: int = 10, max_distance: float = 50.0):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.colors = {}
    
    def _get_color(self, object_id: int) -> Tuple[int, int, int]:
        if object_id not in self.colors:
            color_idx = object_id * 30 % 180
            self.colors[object_id] = self.hsv_to_rgb(color_idx, 255, 255)
        return self.colors[object_id]
    
    def hsv_to_rgb(self, h: int, s: int, v: int) -> Tuple[int, int, int]:
        h = h % 180
        hsv_color = np.uint8([[[h, s, v]]])
        rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)
        return (int(rgb_color[0][0][0]), int(rgb_color[0][0][1]), int(rgb_color[0][0][2]))
    
    def register(self, centroid: Tuple[float, float], bbox: Tuple[int, int, int, int], 
                 confidence: float, class_type: str = "face") -> int:
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
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.disappeared:
            del self.disappeared[object_id]
        if object_id in self.colors:
            del self.colors[object_id]
    
    def update(self, detections: List[Dict]) -> Dict[int, Dict]:
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects.copy()
        
        if len(self.objects) == 0:
            for detection in detections:
                self.register(detection['centroid'], detection['bbox'], 
                            detection['confidence'], detection.get('class_type', 'face'))
            return self.objects.copy()
        
        object_ids = list(self.objects.keys())
        object_centroids = [self.objects[oid]['centroid'] for oid in object_ids]
        detection_centroids = [d['centroid'] for d in detections]
        
        distances = np.zeros((len(object_ids), len(detections)))
        for i, obj_centroid in enumerate(object_centroids):
            for j, det_centroid in enumerate(detection_centroids):
                distances[i, j] = math.sqrt(
                    (obj_centroid[0] - det_centroid[0])**2 + 
                    (obj_centroid[1] - det_centroid[1])**2
                )
        
        used_detections = set()
        used_objects = set()
        matches = []
        
        while True:
            if distances.size == 0:
                break
            
            min_idx = np.unravel_index(np.argmin(distances), distances.shape)
            min_distance = distances[min_idx]
            
            if min_distance > self.max_distance:
                break
            
            object_idx, detection_idx = min_idx
            object_id = object_ids[object_idx]
            
            if object_idx not in used_objects and detection_idx not in used_detections:
                matches.append((object_id, detection_idx, min_distance))
                used_objects.add(object_idx)
                used_detections.add(detection_idx)
            
            distances[object_idx, :] = float('inf')
            distances[:, detection_idx] = float('inf')
        
        for object_id, detection_idx, distance in matches:
            detection = detections[detection_idx]
            self.objects[object_id]['centroid'] = detection['centroid']
            self.objects[object_id]['bbox'] = detection['bbox']
            self.objects[object_id]['confidence'] = detection['confidence']
            self.objects[object_id]['last_seen'] = time.time()
            self.objects[object_id]['track_count'] += 1
            self.disappeared[object_id] = 0
        
        for j, detection in enumerate(detections):
            if j not in used_detections:
                self.register(detection['centroid'], detection['bbox'], 
                            detection['confidence'], detection.get('class_type', 'face'))
        
        for i, object_id in enumerate(object_ids):
            if i not in used_objects:
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
        
        return self.objects.copy()


class VideoProcessor:
    def __init__(self, source: Union[int, str] = 0, 
                 confidence_threshold: float = 0.5,
                 enable_tracking: bool = True,
                 enable_motion: bool = True,
                 tracker_max_distance: float = 50.0,
                 tracker_max_disappeared: int = 15,
                 motion_threshold: float = 0.05):
        """
        Initialize VideoProcessor with motion analysis (NO YOLO).
        """
        self.source = source
        self.confidence_threshold = confidence_threshold
        self.running = False
        self.frame_count = 0
        self.fps = 0
        self.last_time = 0
        self.enable_tracking = enable_tracking
        self.enable_motion = enable_motion
        
        # Initialize face detector (OpenCV only - no MediaPipe)
        self.face_detector = FaceDetector(confidence_threshold)
        self.use_face_detection = self.face_detector.available
        
        # Initialize motion detector
        if enable_motion:
            self.motion_detector = MotionDetector(
                high_motion_threshold=motion_threshold,
                min_motion_area=100
            )
            print("✓ Motion detection initialized")
        
        # Initialize tracker
        if enable_tracking and self.use_face_detection:
            self.face_tracker = EuclideanTracker(
                max_disappeared=tracker_max_disappeared,
                max_distance=tracker_max_distance
            )
            print("✓ Object tracking initialized")
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video source: {source}")
        
        # Get video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps_video = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
        
        # Initialize motion detector with frame size
        if enable_motion:
            self.motion_detector.set_frame_size(self.width, self.height)
        
        # Statistics
        self.motion_metrics_history = []
        self.max_metrics_history = 100
        
        print(f"✓ Video source: {source}")
        print(f"✓ Resolution: {self.width}x{self.height}")
        print(f"✓ Face Detection: {'Enabled' if self.use_face_detection else 'Disabled'}")
        print(f"✓ Person Detection: Disabled (YOLO not available)")
        print(f"✓ Tracking: {'Enabled' if (enable_tracking and self.use_face_detection) else 'Disabled'}")
        print(f"✓ Motion Analysis: {'Enabled' if enable_motion else 'Disabled'}")
    
    def get_face_bboxes(self, tracked_faces: Dict) -> List[Tuple]:
        """Get list of face bounding boxes from tracked faces."""
        face_bboxes = []
        if self.enable_tracking and tracked_faces:
            for obj_id, obj_info in tracked_faces.items():
                if self.face_tracker.disappeared.get(obj_id, 0) == 0:
                    face_bboxes.append(obj_info['bbox'])
        return face_bboxes
    
    def analyze_motion(self, frame: np.ndarray, face_bboxes: List[Tuple]) -> Dict[str, Any]:
        """Analyze motion in the frame."""
        if not self.enable_motion:
            return {}
        
        # Calculate motion data
        motion_data = self.motion_detector.calculate_motion_score(frame)
        
        # Check for motion near faces
        motion_near_faces_data = self.motion_detector.check_motion_near_faces(
            motion_data, face_bboxes
        )
        
        # Combine all motion metrics
        motion_metrics = {
            'frame_number': self.frame_count,
            'timestamp': time.time(),
            'motion_data': motion_data,
            'motion_near_faces_data': motion_near_faces_data,
            'rapid_motion': motion_data['rapid_motion'],
            'motion_near_faces': motion_near_faces_data['motion_near_faces'],
            'affected_faces': motion_near_faces_data['affected_faces'],
            'motion_score': motion_data['motion_score'],
            'motion_percentage': motion_data['motion_percentage'],
            'num_motion_regions': motion_data['num_motion_regions']
        }
        
        # Store in history
        self.motion_metrics_history.append(motion_metrics)
        if len(self.motion_metrics_history) > self.max_metrics_history:
            self.motion_metrics_history.pop(0)
        
        return motion_metrics
    
    def draw_tracked_face(self, frame: np.ndarray, object_id: int, obj_info: Dict) -> np.ndarray:
        """Draw a tracked face with its ID and information."""
        if self.enable_tracking:
            color = self.face_tracker._get_color(object_id)
        else:
            color = (0, 255, 0)  # Green for faces
        
        x1, y1, x2, y2 = obj_info['bbox']
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        id_text = f"ID: {object_id}"
        
        (text_width, text_height), _ = cv2.getTextSize(
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
        
        # Draw centroid point
        centroid = obj_info['centroid']
        cv2.circle(frame, (int(centroid[0]), int(centroid[1])), 3, color, -1)
        
        return frame
    
    def draw_motion_ui(self, frame: np.ndarray, motion_metrics: Dict) -> np.ndarray:
        """Draw motion analysis UI on frame."""
        if not self.enable_motion or not motion_metrics:
            return frame
        
        y_pos = 150
        line_height = 25
        
        # Motion score
        motion_score = motion_metrics['motion_score']
        motion_percent = motion_metrics['motion_percentage']
        motion_color = (0, 255, 255)  # Yellow for motion
        
        cv2.putText(frame, f'Motion Score: {motion_percent:.1f}%', 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, motion_color, 2)
        
        # Motion bar visualization
        bar_width = 200
        bar_height = 10
        bar_x = 10
        bar_y = y_pos + 20
        
        # Background bar
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (100, 100, 100), -1)
        
        # Motion level bar
        motion_width = int(motion_score * bar_width)
        bar_color = (0, 255, 255) if not motion_metrics['rapid_motion'] else (0, 0, 255)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + motion_width, bar_y + bar_height), 
                     bar_color, -1)
        
        # Threshold line
        threshold_x = bar_x + int(self.motion_detector.high_motion_threshold * bar_width)
        cv2.line(frame, (threshold_x, bar_y - 5), (threshold_x, bar_y + bar_height + 5), 
                (255, 255, 255), 2)
        
        # Rapid motion indicator
        if motion_metrics['rapid_motion']:
            cv2.putText(frame, 'RAPID MOTION!', 
                       (bar_x + bar_width + 10, bar_y + bar_height),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        y_pos += 50
        
        # Motion near faces indicator
        if motion_metrics['motion_near_faces']:
            cv2.putText(frame, 'MOTION NEAR FACE!', 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            y_pos += line_height
            
            # Show affected face indices
            affected_faces = motion_metrics['affected_faces']
            if affected_faces:
                faces_text = f'Affected Faces: {", ".join(map(str, affected_faces))}'
                cv2.putText(frame, faces_text, 
                           (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                y_pos += line_height
        
        # Motion regions count
        cv2.putText(frame, f'Motion Regions: {motion_metrics["num_motion_regions"]}', 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def draw_fps_ui(self, frame: np.ndarray) -> np.ndarray:
        """Draw FPS and basic UI."""
        fps_text = f'FPS: {self.fps:.1f}'
        cv2.putText(frame, fps_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        stats_text = f'Frame: {self.frame_count}'
        cv2.putText(frame, stats_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Detection status
        y_pos = 90
        face_status = 'Faces: ON' if self.use_face_detection else 'Faces: OFF'
        face_color = (0, 255, 0) if self.use_face_detection else (100, 100, 100)
        cv2.putText(frame, face_status, (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, face_color, 1)
        
        y_pos += 20
        motion_status = f'Motion: {"ON" if self.enable_motion else "OFF"}'
        motion_color = (0, 255, 255) if self.enable_motion else (100, 100, 100)
        cv2.putText(frame, motion_status, (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, motion_color, 1)
        
        y_pos += 20
        track_status = f'Tracking: {"ON" if self.enable_tracking else "OFF"}'
        track_color = (255, 255, 0) if self.enable_tracking else (100, 100, 100)
        cv2.putText(frame, track_status, (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, track_color, 1)
        
        return frame
    
    def calculate_fps(self):
        current_time = time.time()
        if self.last_time > 0:
            self.fps = 1 / (current_time - self.last_time)
        self.last_time = current_time
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Process a single frame with all features."""
        face_detections = []
        tracked_faces = {}
        
        # Face detection
        if self.use_face_detection:
            face_detections = self.face_detector.detect_faces(frame)
        
        # Tracking
        if self.enable_tracking and self.use_face_detection:
            tracked_faces = self.face_tracker.update(face_detections)
        
        # Motion analysis
        face_bboxes = self.get_face_bboxes(tracked_faces)
        motion_metrics = self.analyze_motion(frame, face_bboxes)
        
        # Draw tracked faces
        if self.enable_tracking and self.use_face_detection:
            for obj_id, obj_info in tracked_faces.items():
                if self.face_tracker.disappeared.get(obj_id, 0) == 0:
                    frame = self.draw_tracked_face(frame, obj_id, obj_info)
        else:
            # Draw faces without tracking
            for detection in face_detections:
                x1, y1, x2, y2 = detection['bbox']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw motion visualization
        if self.enable_motion and motion_metrics:
            frame = self.motion_detector.visualize_motion(
                frame, 
                motion_metrics['motion_data'],
                motion_metrics['motion_near_faces_data'],
                face_bboxes
            )
        
        # Draw UI elements
        self.calculate_fps()
        frame = self.draw_fps_ui(frame)
        frame = self.draw_motion_ui(frame, motion_metrics)
        
        # Print motion alerts in console (less frequent)
        if motion_metrics and self.frame_count % 30 == 0:  # Every 30 frames
            if motion_metrics['rapid_motion']:
                print(f"[Frame {self.frame_count}] ALERT: Rapid Motion detected! "
                      f"Score: {motion_metrics['motion_percentage']:.1f}%")
            
            if motion_metrics['motion_near_faces']:
                print(f"[Frame {self.frame_count}] ALERT: Motion detected near "
                      f"face(s) {motion_metrics['affected_faces']}")
        
        self.frame_count += 1
        return frame, motion_metrics
    
    def print_motion_statistics(self):
        """Print motion analysis statistics."""
        if not self.enable_motion:
            print("Motion analysis is disabled.")
            return
        
        print("\n" + "="*50)
        print("MOTION ANALYSIS STATISTICS")
        print("="*50)
        
        if self.motion_metrics_history:
            recent_metrics = self.motion_metrics_history[-10:]  # Last 10 frames
            
            rapid_motion_count = sum(1 for m in recent_metrics if m['rapid_motion'])
            motion_near_faces_count = sum(1 for m in recent_metrics if m['motion_near_faces'])
            
            print(f"Recent frames analyzed: {len(recent_metrics)}")
            print(f"Rapid motion events: {rapid_motion_count}")
            print(f"Motion near faces events: {motion_near_faces_count}")
            
            if recent_metrics:
                avg_motion = np.mean([m['motion_score'] for m in recent_metrics])
                print(f"Average motion score: {avg_motion:.4f} ({avg_motion*100:.1f}%)")
        
        print("="*50)
    
    def run(self):
        """Main processing loop."""
        self.running = True
        print("\n" + "="*50)
        print("PICAM - Motion Analysis System (No YOLO)")
        print("="*50)
        print("Features Implemented:")
        print("  ✓ Feature 2: Frame-by-frame motion analysis")
        print("  ✓ Feature 8: Motion near face detection")
        print("  ✓ Feature 10: Rapid motion detection")
        print("  ✓ Face detection (OpenCV)")
        print("  ✓ Face tracking with IDs")
        print("\nControls:")
        print("  'q' or ESC - Quit")
        print("  's' - Save screenshot")
        print("  'p' - Pause/resume")
        print("  't' - Toggle tracking")
        print("  'm' - Toggle motion analysis")
        print("  'd' - Display motion statistics")
        print("  'v' - Toggle motion visualization")
        print("="*50 + "\n")
        
        paused = False
        window_name = 'Picam - Motion Analysis'
        
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
                    processed_frame, motion_metrics = self.process_frame(frame)
                    
                    # Display frame
                    cv2.imshow(window_name, processed_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:
                    break
                elif key == ord('s'):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"picam_motion_{timestamp}.jpg"
                    cv2.imwrite(filename, processed_frame)
                    print(f"✓ Screenshot saved: {filename}")
                elif key == ord('p'):
                    paused = not paused
                    status = "⏸️ Paused" if paused else "▶️ Resumed"
                    print(f"{status}")
                elif key == ord('t'):
                    if self.use_face_detection:
                        self.enable_tracking = not self.enable_tracking
                        status = "ENABLED" if self.enable_tracking else "DISABLED"
                        print(f"Tracking {status}")
                    else:
                        print("Tracking requires face detection to be enabled")
                elif key == ord('m'):
                    self.enable_motion = not self.enable_motion
                    status = "ENABLED" if self.enable_motion else "DISABLED"
                    print(f"Motion analysis {status}")
                elif key == ord('d'):
                    self.print_motion_statistics()
                elif key == ord('v'):
                    if self.enable_motion:
                        self.motion_detector.visualize = not self.motion_detector.visualize
                        status = "ON" if self.motion_detector.visualize else "OFF"
                        print(f"Motion visualization {status}")
                
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
        
        print(f"\n✅ VideoProcessor stopped")
        print(f"📊 Total frames processed: {self.frame_count}")
        
        # Print final motion statistics
        if self.enable_motion:
            self.print_motion_statistics()


def main():
    """Main function to run the VideoProcessor with motion analysis."""
    print("="*50)
    print("PICAM - Motion Analysis System")
    print("="*50)
    print("This version uses OpenCV only (no MediaPipe, no YOLO)")
    print("Required packages: opencv-python, numpy")
    print("="*50)
    
    try:
        # Create processor with motion analysis only
        processor = VideoProcessor(
            source=0,  # Webcam
            confidence_threshold=0.5,
            enable_tracking=True,
            enable_motion=True,
            tracker_max_distance=50.0,
            tracker_max_disappeared=15,
            motion_threshold=0.05  # 5% of frame for rapid motion
        )
        
        # Run the processor
        processor.run()
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure webcam is connected")
        print("2. Try camera index 1 instead of 0")
        print("3. Check if another app is using the camera")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()