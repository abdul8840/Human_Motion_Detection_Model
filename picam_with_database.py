# picam_with_csv.py
import cv2
import numpy as np
import time
import sys
from typing import Optional, Union, List, Tuple, Dict, Any
from collections import OrderedDict, deque
import math
import json
import os
from datetime import datetime, date
import threading
import csv

class CSVManager:
    """CSV file manager for Picam data logging."""
    
    def __init__(self, output_dir: str = "picam_reports"):
        """
        Initialize CSV file manager.
        
        Args:
            output_dir: Directory to store CSV files
        """
        self.output_dir = output_dir
        self.lock = threading.Lock()
        self.initialize_directories()
        
        # File paths
        today_str = date.today().strftime("%Y%m%d")
        self.detections_file = os.path.join(output_dir, f"detections_{today_str}.csv")
        self.alerts_file = os.path.join(output_dir, f"alerts_{today_str}.csv")
        self.daily_stats_file = os.path.join(output_dir, f"daily_stats_{today_str}.csv")
        self.zone_stats_file = os.path.join(output_dir, f"zone_stats_{today_str}.csv")
        self.motion_events_file = os.path.join(output_dir, f"motion_events_{today_str}.csv")
        
        # Initialize CSV files with headers if they don't exist
        self.initialize_csv_files()
        print(f"✓ CSV reporting initialized: {output_dir}")
        
    def initialize_directories(self):
        """Create necessary directories."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
    def initialize_csv_files(self):
        """Initialize CSV files with headers."""
        with self.lock:
            # Detections file
            if not os.path.exists(self.detections_file):
                with open(self.detections_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'timestamp', 'object_id', 'object_type', 'zone_name',
                        'confidence', 'x', 'y', 'width', 'height', 'motion_score'
                    ])
            
            # Alerts file
            if not os.path.exists(self.alerts_file):
                with open(self.alerts_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'timestamp', 'alert_type', 'severity', 'description',
                        'zone_name', 'object_id', 'duration_seconds', 'resolved',
                        'resolved_at'
                    ])
            
            # DailyStats file
            if not os.path.exists(self.daily_stats_file):
                with open(self.daily_stats_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'date', 'total_faces', 'total_persons', 'total_motion_events',
                        'total_alerts', 'avg_motion_score', 'peak_motion_score',
                        'total_zone_occupancy_time'
                    ])
            
            # ZoneStats file
            if not os.path.exists(self.zone_stats_file):
                with open(self.zone_stats_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'date', 'zone_name', 'zone_type', 'total_entries',
                        'total_occupancy_time', 'avg_occupancy_time',
                        'idle_alerts_count', 'unattended_alerts_count',
                        'max_concurrent_persons'
                    ])
            
            # MotionEvents file
            if not os.path.exists(self.motion_events_file):
                with open(self.motion_events_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'timestamp', 'motion_score', 'motion_percentage',
                        'rapid_motion', 'num_regions', 'frame_number'
                    ])
    
    def log_detection(self, timestamp: datetime, object_id: int, object_type: str, 
                     zone_name: Optional[str] = None, confidence: float = 0.0,
                     bbox: Optional[Tuple[int, int, int, int]] = None,
                     motion_score: float = 0.0):
        """
        Log a detection event to CSV.
        
        Args:
            timestamp: Detection timestamp
            object_id: Unique object ID
            object_type: 'face' or 'person'
            zone_name: Name of zone where detection occurred
            confidence: Detection confidence score
            bbox: Bounding box (x, y, width, height)
            motion_score: Motion score at detection time
        """
        x, y, width, height = None, None, None, None
        if bbox:
            x, y, width, height = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
        
        with self.lock:
            try:
                with open(self.detections_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        timestamp.strftime("%Y-%m-%d %H:%M:%S.%f"),
                        object_id,
                        object_type,
                        zone_name,
                        confidence,
                        x,
                        y,
                        width,
                        height,
                        motion_score
                    ])
                return True
            except Exception as e:
                print(f"❌ Error logging detection to CSV: {e}")
                return False
    
    def log_alert(self, timestamp: datetime, alert_type: str, severity: str, 
                 description: str = "", zone_name: Optional[str] = None,
                 object_id: Optional[int] = None, duration_seconds: float = 0.0,
                 resolved: bool = False, resolved_at: Optional[datetime] = None):
        """
        Log an alert event to CSV.
        
        Args:
            timestamp: Alert timestamp
            alert_type: Type of alert ('idle_staff', 'unattended_station', 'rapid_motion', 'motion_near_face')
            severity: Alert severity ('low', 'medium', 'high', 'critical')
            description: Alert description
            zone_name: Zone where alert occurred
            object_id: Object ID related to alert
            duration_seconds: How long the condition persisted
            resolved: Whether alert is resolved
            resolved_at: When alert was resolved
        """
        with self.lock:
            try:
                with open(self.alerts_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        timestamp.strftime("%Y-%m-%d %H:%M:%S.%f"),
                        alert_type,
                        severity,
                        description,
                        zone_name,
                        object_id,
                        duration_seconds,
                        resolved,
                        resolved_at.strftime("%Y-%m-%d %H:%M:%S.%f") if resolved_at else None
                    ])
                return True
            except Exception as e:
                print(f"❌ Error logging alert to CSV: {e}")
                return False
    
    def log_motion_event(self, timestamp: datetime, motion_score: float, 
                        motion_percentage: float, rapid_motion: bool,
                        num_regions: int = 0, frame_number: int = None):
        """
        Log a motion event to CSV.
        
        Args:
            timestamp: Event timestamp
            motion_score: Motion score (0-1)
            motion_percentage: Motion percentage (0-100)
            rapid_motion: Whether rapid motion was detected
            num_regions: Number of motion regions
            frame_number: Frame number
        """
        with self.lock:
            try:
                with open(self.motion_events_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        timestamp.strftime("%Y-%m-%d %H:%M:%S.%f"),
                        motion_score,
                        motion_percentage,
                        rapid_motion,
                        num_regions,
                        frame_number
                    ])
                return True
            except Exception as e:
                print(f"❌ Error logging motion event to CSV: {e}")
                return False
    
    def update_daily_stats(self, stats_date: date, stats_data: Dict[str, Any]):
        """
        Update daily statistics in CSV.
        
        Args:
            stats_date: Date for statistics
            stats_data: Dictionary with statistics
        """
        with self.lock:
            try:
                # Read existing data
                existing_data = []
                if os.path.exists(self.daily_stats_file):
                    with open(self.daily_stats_file, 'r', newline='', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        existing_data = list(reader)
                
                # Update or add entry
                date_str = stats_date.strftime("%Y-%m-%d")
                found = False
                
                for row in existing_data:
                    if row['date'] == date_str:
                        row.update({
                            'total_faces': stats_data.get('total_faces', row.get('total_faces', 0)),
                            'total_persons': stats_data.get('total_persons', row.get('total_persons', 0)),
                            'total_motion_events': stats_data.get('total_motion_events', row.get('total_motion_events', 0)),
                            'total_alerts': stats_data.get('total_alerts', row.get('total_alerts', 0)),
                            'avg_motion_score': stats_data.get('avg_motion_score', row.get('avg_motion_score', 0)),
                            'peak_motion_score': stats_data.get('peak_motion_score', row.get('peak_motion_score', 0)),
                            'total_zone_occupancy_time': stats_data.get('total_zone_occupancy_time', row.get('total_zone_occupancy_time', 0))
                        })
                        found = True
                        break
                
                if not found:
                    existing_data.append({
                        'date': date_str,
                        'total_faces': stats_data.get('total_faces', 0),
                        'total_persons': stats_data.get('total_persons', 0),
                        'total_motion_events': stats_data.get('total_motion_events', 0),
                        'total_alerts': stats_data.get('total_alerts', 0),
                        'avg_motion_score': stats_data.get('avg_motion_score', 0),
                        'peak_motion_score': stats_data.get('peak_motion_score', 0),
                        'total_zone_occupancy_time': stats_data.get('total_zone_occupancy_time', 0)
                    })
                
                # Write back to file
                with open(self.daily_stats_file, 'w', newline='', encoding='utf-8') as f:
                    fieldnames = ['date', 'total_faces', 'total_persons', 'total_motion_events',
                                 'total_alerts', 'avg_motion_score', 'peak_motion_score',
                                 'total_zone_occupancy_time']
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(existing_data)
                
                return True
            except Exception as e:
                print(f"❌ Error updating daily stats in CSV: {e}")
                return False
    
    def update_zone_stats(self, stats_date: date, zone_name: str, zone_type: str,
                         stats_data: Dict[str, Any]):
        """
        Update zone statistics in CSV.
        
        Args:
            stats_date: Date for statistics
            zone_name: Name of the zone
            zone_type: Type of zone
            stats_data: Dictionary with zone statistics
        """
        with self.lock:
            try:
                # Read existing data
                existing_data = []
                if os.path.exists(self.zone_stats_file):
                    with open(self.zone_stats_file, 'r', newline='', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        existing_data = list(reader)
                
                # Update or add entry
                date_str = stats_date.strftime("%Y-%m-%d")
                found = False
                
                for row in existing_data:
                    if row['date'] == date_str and row['zone_name'] == zone_name:
                        row.update({
                            'zone_type': zone_type,
                            'total_entries': stats_data.get('total_entries', row.get('total_entries', 0)),
                            'total_occupancy_time': stats_data.get('total_occupancy_time', row.get('total_occupancy_time', 0)),
                            'avg_occupancy_time': stats_data.get('avg_occupancy_time', row.get('avg_occupancy_time', 0)),
                            'idle_alerts_count': stats_data.get('idle_alerts_count', row.get('idle_alerts_count', 0)),
                            'unattended_alerts_count': stats_data.get('unattended_alerts_count', row.get('unattended_alerts_count', 0)),
                            'max_concurrent_persons': stats_data.get('max_concurrent_persons', row.get('max_concurrent_persons', 0))
                        })
                        found = True
                        break
                
                if not found:
                    existing_data.append({
                        'date': date_str,
                        'zone_name': zone_name,
                        'zone_type': zone_type,
                        'total_entries': stats_data.get('total_entries', 0),
                        'total_occupancy_time': stats_data.get('total_occupancy_time', 0),
                        'avg_occupancy_time': stats_data.get('avg_occupancy_time', 0),
                        'idle_alerts_count': stats_data.get('idle_alerts_count', 0),
                        'unattended_alerts_count': stats_data.get('unattended_alerts_count', 0),
                        'max_concurrent_persons': stats_data.get('max_concurrent_persons', 0)
                    })
                
                # Write back to file
                with open(self.zone_stats_file, 'w', newline='', encoding='utf-8') as f:
                    fieldnames = ['date', 'zone_name', 'zone_type', 'total_entries',
                                 'total_occupancy_time', 'avg_occupancy_time',
                                 'idle_alerts_count', 'unattended_alerts_count',
                                 'max_concurrent_persons']
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(existing_data)
                
                return True
            except Exception as e:
                print(f"❌ Error updating zone stats in CSV: {e}")
                return False
    
    def get_todays_stats(self) -> Dict[str, Any]:
        """Get statistics for today from CSV files."""
        today = date.today()
        today_str = today.strftime("%Y-%m-%d")
        
        result = {
            'daily_stats': {},
            'zone_stats': [],
            'active_alerts': 0,
            'recent_detections': []
        }
        
        try:
            # Read daily stats
            if os.path.exists(self.daily_stats_file):
                with open(self.daily_stats_file, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row['date'] == today_str:
                            result['daily_stats'] = row
                            break
            
            # Read zone stats
            if os.path.exists(self.zone_stats_file):
                with open(self.zone_stats_file, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row['date'] == today_str:
                            result['zone_stats'].append(row)
            
            # Count active alerts (not resolved)
            if os.path.exists(self.alerts_file):
                with open(self.alerts_file, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row['timestamp'].startswith(today_str) and row['resolved'] == 'False':
                            result['active_alerts'] += 1
            
            # Get recent detections (last 10)
            if os.path.exists(self.detections_file):
                with open(self.detections_file, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    detections = list(reader)
                    # Filter today's detections and get last 10
                    today_detections = [d for d in detections if d['timestamp'].startswith(today_str)]
                    result['recent_detections'] = today_detections[-10:] if today_detections else []
            
        except Exception as e:
            print(f"❌ Error reading CSV stats: {e}")
        
        return result
    
    def export_all_data(self, output_dir: str = "exports"):
        """Export all CSV data to a single directory (already in CSV format)."""
        import shutil
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        try:
            # List all CSV files in the reports directory
            csv_files = [f for f in os.listdir(self.output_dir) if f.endswith('.csv')]
            
            for csv_file in csv_files:
                src = os.path.join(self.output_dir, csv_file)
                dst = os.path.join(output_dir, csv_file)
                shutil.copy2(src, dst)
            
            print(f"✓ All CSV files exported to {output_dir}")
            return True
        except Exception as e:
            print(f"❌ Error exporting CSV files: {e}")
            return False
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Remove CSV files older than specified days."""
        try:
            cutoff_date = date.today().replace(day=date.today().day - days_to_keep)
            files_deleted = 0
            
            for filename in os.listdir(self.output_dir):
                if filename.endswith('.csv'):
                    # Extract date from filename
                    date_part = filename.split('_')[-1].replace('.csv', '')
                    try:
                        file_date = datetime.strptime(date_part, "%Y%m%d").date()
                        if file_date < cutoff_date:
                            filepath = os.path.join(self.output_dir, filename)
                            os.remove(filepath)
                            files_deleted += 1
                    except ValueError:
                        continue
            
            print(f"Cleaned up {files_deleted} old CSV files")
            return True
        except Exception as e:
            print(f"❌ Error cleaning up old CSV files: {e}")
            return False
    
    def print_csv_summary(self):
        """Print summary of CSV files."""
        print("\n" + "="*50)
        print("CSV FILES SUMMARY")
        print("="*50)
        
        files = {
            'Detections': self.detections_file,
            'Alerts': self.alerts_file,
            'Daily Stats': self.daily_stats_file,
            'Zone Stats': self.zone_stats_file,
            'Motion Events': self.motion_events_file
        }
        
        for name, filepath in files.items():
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r', newline='', encoding='utf-8') as f:
                        lines = sum(1 for _ in f) - 1  # Subtract header
                    print(f"{name}: {lines} records")
                except:
                    print(f"{name}: Error reading file")
            else:
                print(f"{name}: File not found")
        
        print("="*50)


class Zone:
    """Define a zone (region of interest) with custom logic."""
    def __init__(self, name: str, points: List[Tuple[int, int]], 
                 zone_type: str = "counter", idle_threshold: float = 120.0,
                 unattended_threshold: float = 300.0, motion_threshold: float = 0.01):
        self.name = name
        self.points = np.array(points, dtype=np.int32)
        self.zone_type = zone_type
        self.idle_threshold = idle_threshold
        self.unattended_threshold = unattended_threshold
        self.motion_threshold = motion_threshold
        
        # Calculate zone properties
        self.bbox = self.calculate_bounding_box()
        self.area = self.calculate_area()
        self.center = self.calculate_center()
        
        # Zone state
        self.person_data = {}
        self.zone_occupied = False
        self.last_occupied_time = None
        self.last_vacant_time = time.time()
        
        # Alerts
        self.idle_alert_active = False
        self.unattended_alert_active = False
        self.idle_start_time = None
        self.unattended_start_time = None
        
        # Statistics
        self.total_occupancy_time = 0.0
        self.entry_count = 0
        self.max_concurrent_persons = 0
        self.idle_alerts_count = 0
        self.unattended_alerts_count = 0
        
    def calculate_bounding_box(self) -> Tuple[int, int, int, int]:
        x_coords = [p[0] for p in self.points]
        y_coords = [p[1] for p in self.points]
        return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
    
    def calculate_area(self) -> float:
        x = self.points[:, 0]
        y = self.points[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    
    def calculate_center(self) -> Tuple[int, int]:
        x_coords = [p[0] for p in self.points]
        y_coords = [p[1] for p in self.points]
        return ((min(x_coords) + max(x_coords)) // 2, (min(y_coords) + max(y_coords)) // 2)
    
    def is_point_in_zone(self, point: Tuple[float, float]) -> bool:
        x, y = point
        n = len(self.points)
        inside = False
        
        p1x, p1y = self.points[0]
        for i in range(n + 1):
            p2x, p2y = self.points[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def is_bbox_in_zone(self, bbox: Tuple[int, int, int, int]) -> bool:
        x1, y1, x2, y2 = bbox
        centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
        if self.is_point_in_zone(centroid):
            return True
        
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        for corner in corners:
            if self.is_point_in_zone(corner):
                return True
        
        return False
    
    def update_person(self, person_id: int, bbox: Tuple[int, int, int, int], 
                     motion_score: float = 0.0, current_time: float = None):
        if current_time is None:
            current_time = time.time()
        
        if person_id not in self.person_data:
            self.person_data[person_id] = {
                'entry_time': current_time,
                'last_motion_time': current_time,
                'last_update_time': current_time,
                'motion_history': deque(maxlen=30),
                'total_motion': 0.0,
                'motion_samples': 0,
                'bbox': bbox
            }
            self.entry_count += 1
            
            # Update max concurrent persons
            current_count = len(self.person_data)
            if current_count > self.max_concurrent_persons:
                self.max_concurrent_persons = current_count
        else:
            self.person_data[person_id]['last_update_time'] = current_time
            self.person_data[person_id]['bbox'] = bbox
            
            if motion_score > 0:
                self.person_data[person_id]['last_motion_time'] = current_time
                self.person_data[person_id]['motion_history'].append(motion_score)
                self.person_data[person_id]['total_motion'] += motion_score
                self.person_data[person_id]['motion_samples'] += 1
    
    def remove_person(self, person_id: int, current_time: float = None):
        if person_id in self.person_data:
            if current_time is None:
                current_time = time.time()
            
            entry_time = self.person_data[person_id]['entry_time']
            occupancy_time = current_time - entry_time
            self.total_occupancy_time += occupancy_time
            
            del self.person_data[person_id]
    
    def check_idle_persons(self, current_time: float = None) -> List[int]:
        if current_time is None:
            current_time = time.time()
        
        idle_persons = []
        
        for person_id, data in self.person_data.items():
            time_since_motion = current_time - data['last_motion_time']
            avg_motion = 0.0
            if data['motion_samples'] > 0:
                avg_motion = data['total_motion'] / data['motion_samples']
            
            if (time_since_motion > self.idle_threshold and 
                avg_motion < self.motion_threshold):
                idle_persons.append(person_id)
        
        return idle_persons
    
    def check_zone_status(self, current_time: float = None) -> Dict[str, Any]:
        if current_time is None:
            current_time = time.time()
        
        was_occupied = self.zone_occupied
        self.zone_occupied = len(self.person_data) > 0
        
        if self.zone_occupied:
            self.last_occupied_time = current_time
        else:
            self.last_vacant_time = current_time
        
        idle_persons = self.check_idle_persons(current_time)
        
        unattended_time = 0.0
        if not self.zone_occupied:
            unattended_time = current_time - self.last_occupied_time if self.last_occupied_time else float('inf')
        
        status_changed = False
        
        # Idle alert
        if idle_persons and not self.idle_alert_active:
            self.idle_alert_active = True
            self.idle_start_time = current_time
            self.idle_alerts_count += 1
            status_changed = True
        elif not idle_persons and self.idle_alert_active:
            self.idle_alert_active = False
            self.idle_start_time = None
            status_changed = True
        
        # Unattended alert
        if (unattended_time > self.unattended_threshold and 
            not self.unattended_alert_active):
            self.unattended_alert_active = True
            self.unattended_start_time = current_time
            self.unattended_alerts_count += 1
            status_changed = True
        elif (unattended_time <= self.unattended_threshold and 
              self.unattended_alert_active):
            self.unattended_alert_active = False
            self.unattended_start_time = None
            status_changed = True
        
        return {
            'zone_name': self.name,
            'zone_type': self.zone_type,
            'occupied': self.zone_occupied,
            'person_count': len(self.person_data),
            'person_ids': list(self.person_data.keys()),
            'idle_persons': idle_persons,
            'idle_alert': self.idle_alert_active,
            'idle_duration': current_time - self.idle_start_time if self.idle_alert_active else 0,
            'unattended_time': unattended_time if not self.zone_occupied else 0,
            'unattended_alert': self.unattended_alert_active,
            'unattended_duration': current_time - self.unattended_start_time if self.unattended_alert_active else 0,
            'area': self.area,
            'center': self.center,
            'status_changed': status_changed
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'type': self.zone_type,
            'total_entries': self.entry_count,
            'total_occupancy_time': self.total_occupancy_time,
            'avg_occupancy_time': self.total_occupancy_time / self.entry_count if self.entry_count > 0 else 0,
            'currently_occupied': self.zone_occupied,
            'current_persons': len(self.person_data),
            'max_concurrent_persons': self.max_concurrent_persons,
            'idle_alert': self.idle_alert_active,
            'unattended_alert': self.unattended_alert_active,
            'idle_alerts_count': self.idle_alerts_count,
            'unattended_alerts_count': self.unattended_alerts_count
        }


class MotionDetector:
    """Motion detection using background subtraction."""
    def __init__(self, history: int = 500, var_threshold: int = 16, 
                 detect_shadows: bool = True, high_motion_threshold: float = 0.05,
                 min_motion_area: int = 100):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows
        )
        self.high_motion_threshold = high_motion_threshold
        self.min_motion_area = min_motion_area
        self.motion_history = []
        self.max_history = 30
        self.frame_width = 0
        self.frame_height = 0
        self.total_pixels = 0
        
    def set_frame_size(self, width: int, height: int):
        self.frame_width = width
        self.frame_height = height
        self.total_pixels = width * height
    
    def calculate_motion_score(self, frame: np.ndarray) -> Dict[str, Any]:
        fg_mask = self.bg_subtractor.apply(frame)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_areas = []
        motion_centroids = []
        valid_contours = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_motion_area:
                motion_areas.append(area)
                valid_contours.append(contour)
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    motion_centroids.append((cx, cy))
        
        total_motion_area = sum(motion_areas)
        motion_score = total_motion_area / self.total_pixels if self.total_pixels > 0 else 0.0
        rapid_motion = motion_score > self.high_motion_threshold
        
        self.motion_history.append(motion_score)
        if len(self.motion_history) > self.max_history:
            self.motion_history.pop(0)
        
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


class PersonDetector:
    """Simple person detection using motion and size heuristics."""
    def __init__(self, min_person_area: int = 1000, max_person_area: int = 50000):
        self.min_person_area = min_person_area
        self.max_person_area = max_person_area
        
    def detect_from_motion(self, motion_data: Dict, frame_shape: Tuple[int, int]) -> List[Dict]:
        persons = []
        
        for contour in motion_data['contours']:
            area = cv2.contourArea(contour)
            
            if self.min_person_area < area < self.max_person_area:
                x, y, w, h = cv2.boundingRect(contour)
                bbox = (x, y, x + w, y + h)
                centroid = (x + w // 2, y + h // 2)
                
                persons.append({
                    'bbox': bbox,
                    'centroid': centroid,
                    'area': area,
                    'confidence': min(1.0, area / self.max_person_area)
                })
        
        return persons


class VideoProcessor:
    def __init__(self, source: Union[int, str] = 0, 
                 confidence_threshold: float = 0.5,
                 enable_zones: bool = True,
                 enable_motion: bool = True,
                 enable_person_detection: bool = True,
                 enable_csv_logging: bool = True,
                 motion_threshold: float = 0.05):
        """
        Initialize VideoProcessor with CSV logging.
        """
        self.source = source
        self.confidence_threshold = confidence_threshold
        self.running = False
        self.frame_count = 0
        self.fps = 0
        self.last_time = 0
        self.enable_zones = enable_zones
        self.enable_motion = enable_motion
        self.enable_person_detection = enable_person_detection
        self.enable_csv_logging = enable_csv_logging
        
        # Initialize CSV manager
        if enable_csv_logging:
            self.csv_manager = CSVManager("picam_reports")
            print("✓ CSV logging initialized")
        
        # Initialize video capture FIRST
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video source: {source}")
        
        # Get video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps_video = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
        
        # Initialize motion detector
        if enable_motion:
            self.motion_detector = MotionDetector(
                high_motion_threshold=motion_threshold,
                min_motion_area=100
            )
            self.motion_detector.set_frame_size(self.width, self.height)
            print("✓ Motion detection initialized")
        
        # Initialize person detector
        if enable_person_detection:
            self.person_detector = PersonDetector()
            print("✓ Person detection initialized")
        
        # Initialize zones
        self.zones = []
        self.zone_config_file = "zones_config.json"
        if enable_zones:
            self.load_zones_from_file()
            print(f"✓ Zone system initialized: {len(self.zones)} zones")
        
        # Statistics
        self.stats = {
            'total_faces': 0,
            'total_persons': 0,
            'total_motion_events': 0,
            'total_alerts': 0,
            'motion_scores': [],
            'peak_motion_score': 0,
            'total_zone_occupancy_time': 0
        }
        
        # Alert tracking
        self.active_alerts = {}
        
        print(f"✓ Video source: {source}")
        print(f"✓ Resolution: {self.width}x{self.height}")
        print(f"✓ Zones: {'Enabled' if enable_zones else 'Disabled'}")
        print(f"✓ Motion Analysis: {'Enabled' if enable_motion else 'Disabled'}")
        print(f"✓ Person Detection: {'Enabled' if enable_person_detection else 'Disabled'}")
        print(f"✓ CSV Logging: {'Enabled' if enable_csv_logging else 'Disabled'}")
    
    def create_default_zones(self):
        """Create default zones."""
        counter_points = [
            (self.width // 2 - 200, self.height - 150),
            (self.width // 2 + 200, self.height - 150),
            (self.width // 2 + 200, self.height - 50),
            (self.width // 2 - 200, self.height - 50)
        ]
        
        table_points = [
            (self.width // 2 - 150, self.height // 2 - 100),
            (self.width // 2 + 150, self.height // 2 - 100),
            (self.width // 2 + 150, self.height // 2 + 100),
            (self.width // 2 - 150, self.height // 2 + 100)
        ]
        
        entrance_points = [
            (50, 100),
            (250, 100),
            (250, 300),
            (50, 300)
        ]
        
        self.zones = [
            Zone("Counter", counter_points, "counter", 
                 idle_threshold=120.0, unattended_threshold=300.0),
            Zone("Table", table_points, "table",
                 idle_threshold=180.0, unattended_threshold=600.0),
            Zone("Entrance", entrance_points, "entrance",
                 idle_threshold=60.0, unattended_threshold=0.0)
        ]
        
        self.save_zones_to_file()
    
    def save_zones_to_file(self):
        zones_data = []
        for zone in self.zones:
            zones_data.append({
                'name': zone.name,
                'points': zone.points.tolist(),
                'zone_type': zone.zone_type,
                'idle_threshold': zone.idle_threshold,
                'unattended_threshold': zone.unattended_threshold,
                'motion_threshold': zone.motion_threshold
            })
        
        with open(self.zone_config_file, 'w') as f:
            json.dump(zones_data, f, indent=2)
        print(f"✓ Zones saved to {self.zone_config_file}")
    
    def load_zones_from_file(self):
        if os.path.exists(self.zone_config_file):
            try:
                with open(self.zone_config_file, 'r') as f:
                    zones_data = json.load(f)
                
                self.zones = []
                for zone_data in zones_data:
                    zone = Zone(
                        name=zone_data['name'],
                        points=zone_data['points'],
                        zone_type=zone_data['zone_type'],
                        idle_threshold=zone_data.get('idle_threshold', 120.0),
                        unattended_threshold=zone_data.get('unattended_threshold', 300.0),
                        motion_threshold=zone_data.get('motion_threshold', 0.01)
                    )
                    self.zones.append(zone)
                print(f"✓ Loaded {len(self.zones)} zones from {self.zone_config_file}")
            except Exception as e:
                print(f"⚠ Error loading zones: {e}")
                print("Creating default zones...")
                self.create_default_zones()
        else:
            print("⚠ No zones config file found. Creating default zones...")
            self.create_default_zones()
    
    def log_detection_event(self, object_id: int, object_type: str, 
                           bbox: Tuple[int, int, int, int], confidence: float,
                           zone_name: Optional[str] = None, motion_score: float = 0.0):
        """Log detection to CSV."""
        if not self.enable_csv_logging:
            return
        
        timestamp = datetime.now()
        
        # Log detection
        success = self.csv_manager.log_detection(
            timestamp=timestamp,
            object_id=object_id,
            object_type=object_type,
            zone_name=zone_name,
            confidence=confidence,
            bbox=bbox,
            motion_score=motion_score
        )
        
        if success:
            # Update statistics
            if object_type == 'face':
                self.stats['total_faces'] += 1
            elif object_type == 'person':
                self.stats['total_persons'] += 1
    
    def log_alert_event(self, alert_type: str, severity: str, description: str,
                       zone_name: Optional[str] = None, object_id: Optional[int] = None,
                       duration_seconds: float = 0.0):
        """Log alert to CSV."""
        if not self.enable_csv_logging:
            return
        
        timestamp = datetime.now()
        alert_key = f"{alert_type}_{zone_name}_{object_id}"
        
        # Log alert
        success = self.csv_manager.log_alert(
            timestamp=timestamp,
            alert_type=alert_type,
            severity=severity,
            description=description,
            zone_name=zone_name,
            object_id=object_id,
            duration_seconds=duration_seconds
        )
        
        if success:
            self.stats['total_alerts'] += 1
            self.active_alerts[alert_key] = timestamp
    
    def log_motion_event(self, motion_data: Dict):
        """Log motion event to CSV."""
        if not self.enable_csv_logging:
            return
        
        timestamp = datetime.now()
        
        # Log motion event
        success = self.csv_manager.log_motion_event(
            timestamp=timestamp,
            motion_score=motion_data['motion_score'],
            motion_percentage=motion_data['motion_percentage'],
            rapid_motion=motion_data['rapid_motion'],
            num_regions=motion_data['num_motion_regions'],
            frame_number=self.frame_count
        )
        
        if success:
            self.stats['total_motion_events'] += 1
            self.stats['motion_scores'].append(motion_data['motion_score'])
            
            # Update peak motion score
            if motion_data['motion_score'] > self.stats['peak_motion_score']:
                self.stats['peak_motion_score'] = motion_data['motion_score']
            
            # Log rapid motion alert if detected
            if motion_data['rapid_motion']:
                self.log_alert_event(
                    alert_type='rapid_motion',
                    severity='high',
                    description=f"Rapid motion detected: {motion_data['motion_percentage']:.1f}%",
                    duration_seconds=0.0
                )
    
    def update_csv_stats(self):
        """Update daily statistics in CSV."""
        if not self.enable_csv_logging:
            return
        
        today = date.today()
        
        # Calculate average motion score
        avg_motion_score = 0
        if self.stats['motion_scores']:
            avg_motion_score = np.mean(self.stats['motion_scores'])
        
        # Calculate total zone occupancy time
        total_zone_occupancy = sum(zone.total_occupancy_time for zone in self.zones)
        self.stats['total_zone_occupancy_time'] = total_zone_occupancy
        
        # Update daily stats
        self.csv_manager.update_daily_stats(today, {
            'total_faces': self.stats['total_faces'],
            'total_persons': self.stats['total_persons'],
            'total_motion_events': self.stats['total_motion_events'],
            'total_alerts': self.stats['total_alerts'],
            'avg_motion_score': avg_motion_score,
            'peak_motion_score': self.stats['peak_motion_score'],
            'total_zone_occupancy_time': total_zone_occupancy
        })
        
        # Update zone stats
        for zone in self.zones:
            zone_stats = zone.get_statistics()
            self.csv_manager.update_zone_stats(
                stats_date=today,
                zone_name=zone.name,
                zone_type=zone.zone_type,
                stats_data={
                    'total_entries': zone_stats['total_entries'],
                    'total_occupancy_time': zone_stats['total_occupancy_time'],
                    'avg_occupancy_time': zone_stats['avg_occupancy_time'],
                    'idle_alerts_count': zone_stats['idle_alerts_count'],
                    'unattended_alerts_count': zone_stats['unattended_alerts_count'],
                    'max_concurrent_persons': zone_stats['max_concurrent_persons']
                }
            )
    
    def process_zones(self, persons: List[Dict], motion_score: float = 0.0) -> List[Dict]:
        if not self.enable_zones:
            return []
        
        current_time = time.time()
        zone_statuses = []
        
        # Remove timed out persons
        for zone in self.zones:
            persons_to_remove = []
            for person_id in zone.person_data.keys():
                data = zone.person_data[person_id]
                if current_time - data['last_update_time'] > 1.0:
                    persons_to_remove.append(person_id)
            
            for person_id in persons_to_remove:
                zone.remove_person(person_id, current_time)
        
        # Update zones with current persons
        for person_idx, person in enumerate(persons):
            person_id = person_idx + 1
            
            # Log person detection
            self.log_detection_event(
                object_id=person_id,
                object_type='person',
                bbox=person['bbox'],
                confidence=person['confidence'],
                motion_score=motion_score
            )
            
            # Check which zones person is in
            for zone in self.zones:
                if zone.is_bbox_in_zone(person['bbox']):
                    zone.update_person(person_id, person['bbox'], motion_score, current_time)
                    
                    # Log zone-specific detection
                    self.log_detection_event(
                        object_id=person_id,
                        object_type='person',
                        bbox=person['bbox'],
                        confidence=person['confidence'],
                        zone_name=zone.name,
                        motion_score=motion_score
                    )
        
        # Check zone statuses and log alerts
        for zone in self.zones:
            zone_status = zone.check_zone_status(current_time)
            zone_statuses.append(zone_status)
            
            # Log alerts if status changed
            if zone_status['status_changed']:
                if zone_status['idle_alert']:
                    self.log_alert_event(
                        alert_type='idle_staff',
                        severity='medium',
                        description=f"Person(s) {zone_status['idle_persons']} idle in {zone.name}",
                        zone_name=zone.name,
                        object_id=zone_status['idle_persons'][0] if zone_status['idle_persons'] else None,
                        duration_seconds=zone_status['idle_duration']
                    )
                
                if zone_status['unattended_alert']:
                    self.log_alert_event(
                        alert_type='unattended_station',
                        severity='high',
                        description=f"{zone.name} unattended for {zone_status['unattended_duration']:.1f}s",
                        zone_name=zone.name,
                        duration_seconds=zone_status['unattended_duration']
                    )
        
        return zone_statuses
    
    def draw_zones(self, frame: np.ndarray) -> np.ndarray:
        for zone in self.zones:
            zone_color = (0, 165, 255)
            if zone.zone_type == "counter":
                zone_color = (255, 0, 255)
            elif zone.zone_type == "table":
                zone_color = (0, 255, 255)
            elif zone.zone_type == "entrance":
                zone_color = (0, 255, 0)
            
            overlay = frame.copy()
            cv2.fillPoly(overlay, [zone.points], zone_color)
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
            
            cv2.polylines(frame, [zone.points], True, zone_color, 2)
            
            cv2.putText(frame, zone.name, (zone.center[0] - 30, zone.center[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            person_count = len(zone.person_data)
            count_text = f"Persons: {person_count}"
            cv2.putText(frame, count_text, (zone.center[0] - 30, zone.center[1] + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if zone.idle_alert_active:
                cv2.putText(frame, "IDLE!", (zone.bbox[0], zone.bbox[1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if zone.unattended_alert_active:
                cv2.putText(frame, "UNATTENDED!", (zone.bbox[0], zone.bbox[1] - 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame
    
    def draw_persons(self, frame: np.ndarray, persons: List[Dict]) -> np.ndarray:
        for idx, person in enumerate(persons):
            person_id = idx + 1
            x1, y1, x2, y2 = person['bbox']
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"P{person_id}", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            centroid = person['centroid']
            cv2.circle(frame, (int(centroid[0]), int(centroid[1])), 3, (255, 0, 0), -1)
        
        return frame
    
    def draw_ui(self, frame: np.ndarray, motion_metrics: Dict, zone_statuses: List[Dict]) -> np.ndarray:
        fps_text = f'FPS: {self.fps:.1f}'
        cv2.putText(frame, fps_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        frame_text = f'Frame: {self.frame_count}'
        cv2.putText(frame, frame_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if motion_metrics:
            motion_text = f'Motion: {motion_metrics["motion_percentage"]:.1f}%'
            cv2.putText(frame, motion_text, (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        y_pos = 120
        idle_alerts = 0
        unattended_alerts = 0
        
        for zone_status in zone_statuses:
            if zone_status['idle_alert']:
                idle_alerts += 1
            if zone_status['unattended_alert']:
                unattended_alerts += 1
        
        if idle_alerts > 0:
            cv2.putText(frame, f'Idle Alerts: {idle_alerts}', (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y_pos += 25
        
        if unattended_alerts > 0:
            cv2.putText(frame, f'Unattended Alerts: {unattended_alerts}', (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y_pos += 25
        
        # CSV stats
        if self.enable_csv_logging:
            csv_stats = self.csv_manager.get_todays_stats()
            active_alerts = csv_stats.get('active_alerts', 0)
            cv2.putText(frame, f'Active Alerts: {active_alerts}', (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            y_pos += 20
            
            total_detections = self.stats['total_faces'] + self.stats['total_persons']
            cv2.putText(frame, f'Detections: {total_detections}', (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            y_pos += 20
            
            csv_text = "CSV: ACTIVE"
            cv2.putText(frame, csv_text, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return frame
    
    def calculate_fps(self):
        current_time = time.time()
        if self.last_time > 0:
            self.fps = 1 / (current_time - self.last_time)
        self.last_time = current_time
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict, List[Dict]]:
        motion_metrics = {}
        persons = []
        zone_statuses = []
        
        # Motion detection
        if self.enable_motion:
            motion_metrics = self.motion_detector.calculate_motion_score(frame)
            self.log_motion_event(motion_metrics)
        
        # Person detection
        if self.enable_person_detection and motion_metrics:
            persons = self.person_detector.detect_from_motion(motion_metrics, frame.shape[:2])
        
        # Zone processing
        if self.enable_zones:
            motion_score = motion_metrics.get('motion_score', 0.0) if motion_metrics else 0.0
            zone_statuses = self.process_zones(persons, motion_score)
        
        # Draw everything
        if self.enable_zones:
            frame = self.draw_zones(frame)
        
        if self.enable_person_detection:
            frame = self.draw_persons(frame, persons)
        
        # Draw UI
        self.calculate_fps()
        frame = self.draw_ui(frame, motion_metrics, zone_statuses)
        
        # Update CSV stats every 60 frames (about 2 seconds at 30 FPS)
        if self.frame_count % 60 == 0 and self.enable_csv_logging:
            self.update_csv_stats()
        
        self.frame_count += 1
        return frame, motion_metrics, zone_statuses
    
    def print_csv_stats(self):
        """Print CSV file statistics."""
        if not self.enable_csv_logging:
            print("CSV logging is disabled.")
            return
        
        print("\n" + "="*50)
        print("CSV FILE STATISTICS")
        print("="*50)
        
        stats = self.csv_manager.get_todays_stats()
        
        print(f"\nToday's Statistics ({date.today()}):")
        if stats['daily_stats']:
            daily = stats['daily_stats']
            print(f"  Total Faces: {daily.get('total_faces', 0)}")
            print(f"  Total Persons: {daily.get('total_persons', 0)}")
            print(f"  Total Motion Events: {daily.get('total_motion_events', 0)}")
            print(f"  Total Alerts: {daily.get('total_alerts', 0)}")
            print(f"  Active Alerts: {stats.get('active_alerts', 0)}")
            print(f"  Avg Motion Score: {daily.get('avg_motion_score', 0):.3f}")
            print(f"  Peak Motion Score: {daily.get('peak_motion_score', 0):.3f}")
        
        if stats['zone_stats']:
            print(f"\nZone Statistics:")
            for zone in stats['zone_stats']:
                print(f"  {zone['zone_name']} ({zone['zone_type']}):")
                print(f"    Entries: {zone['total_entries']}")
                print(f"    Occupancy Time: {zone['total_occupancy_time']:.1f}s")
                print(f"    Idle Alerts: {zone['idle_alerts_count']}")
                print(f"    Unattended Alerts: {zone['unattended_alerts_count']}")
        
        # Print CSV file summary
        self.csv_manager.print_csv_summary()
        
        print("="*50)
    
    def run(self):
        """Main processing loop."""
        self.running = True
        print("\n" + "="*50)
        print("PICAM - Complete System with CSV Logging")
        print("="*50)
        print("Features Implemented:")
        print("  ✓ Face/Person detection with tracking")
        print("  ✓ Motion analysis (rapid motion, motion near faces)")
        print("  ✓ Zone monitoring (idle staff, unattended stations)")
        print("  ✓ CSV file logging (no database required)")
        print("\nCSV Files Created:")
        print("  ✓ detections_YYYYMMDD.csv - All detection events")
        print("  ✓ alerts_YYYYMMDD.csv - All alert events")
        print("  ✓ daily_stats_YYYYMMDD.csv - Daily statistics")
        print("  ✓ zone_stats_YYYYMMDD.csv - Zone-specific statistics")
        print("  ✓ motion_events_YYYYMMDD.csv - Motion analysis events")
        print("\nFiles are saved in 'picam_reports' folder")
        print("\nControls:")
        print("  'q' or ESC - Quit")
        print("  's' - Save screenshot")
        print("  'p' - Pause/resume")
        print("  'd' - Display CSV statistics")
        print("  'e' - Export all CSV files")
        print("  'c' - Cleanup old CSV files (30+ days)")
        print("  'z' - Toggle zones")
        print("  'm' - Toggle motion detection")
        print("  'b' - Toggle CSV logging")
        print("="*50 + "\n")
        
        paused = False
        window_name = 'Picam - Complete System (CSV)'
        
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
                    processed_frame, motion_metrics, zone_statuses = self.process_frame(frame)
                    
                    # Display frame
                    cv2.imshow(window_name, processed_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:
                    break
                elif key == ord('s'):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"picam_csv_{timestamp}.jpg"
                    cv2.imwrite(filename, processed_frame)
                    print(f"✓ Screenshot saved: {filename}")
                elif key == ord('p'):
                    paused = not paused
                    status = "⏸️ Paused" if paused else "▶️ Resumed"
                    print(f"{status}")
                elif key == ord('d'):
                    self.print_csv_stats()
                elif key == ord('e'):
                    if self.enable_csv_logging:
                        self.csv_manager.export_all_data("csv_exports")
                        print("✓ All CSV files exported to 'csv_exports' folder")
                elif key == ord('c'):
                    if self.enable_csv_logging:
                        self.csv_manager.cleanup_old_data(days_to_keep=30)
                        print("✓ Old CSV files cleaned up")
                elif key == ord('z'):
                    self.enable_zones = not self.enable_zones
                    status = "ENABLED" if self.enable_zones else "DISABLED"
                    print(f"Zones {status}")
                elif key == ord('m'):
                    self.enable_motion = not self.enable_motion
                    status = "ENABLED" if self.enable_motion else "DISABLED"
                    print(f"Motion detection {status}")
                elif key == ord('b'):
                    self.enable_csv_logging = not self.enable_csv_logging
                    status = "ENABLED" if self.enable_csv_logging else "DISABLED"
                    print(f"CSV logging {status}")
                
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
        
        # Final CSV update
        if self.enable_csv_logging:
            self.update_csv_stats()
        
        print(f"\n✅ VideoProcessor stopped")
        print(f"📊 Total frames processed: {self.frame_count}")
        
        # Print final statistics
        self.print_csv_stats()


def main():
    """Main function to run the complete Picam system with CSV logging."""
    print("="*50)
    print("PICAM - Complete Computer Vision System")
    print("="*50)
    print("This version uses CSV files instead of a database.")
    print("Required packages: opencv-python, numpy")
    print("="*50)
    
    try:
        # Create processor with all features enabled
        processor = VideoProcessor(
            source=0,  # Webcam
            confidence_threshold=0.5,
            enable_zones=True,
            enable_motion=True,
            enable_person_detection=True,
            enable_csv_logging=True,
            motion_threshold=0.05
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