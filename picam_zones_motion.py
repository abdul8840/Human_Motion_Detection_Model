# picam_zones_motion_fixed.py  (Session-5 ready – just replace the whole file with this one)
# ------------------------------------------------------------------
# All previous functionality is preserved.
# NEW: SQLite logging for Detections, Alerts, DailyStats.
# ------------------------------------------------------------------
import cv2
import numpy as np
import time
import sys
import math
import json
import os
import sqlite3
import datetime
from pathlib import Path
from typing import Optional, Union, List, Tuple, Dict, Any
from collections import OrderedDict, deque

# ------------------------------------------------------
# 1.  DATABASE MANAGER  (NEW)
# ------------------------------------------------------
class DatabaseManager:
    """
    Lightweight SQLite helper that automatically creates the schema
    and exposes simple insert methods for Detections, Alerts & DailyStats.
    """
    DB_FILE = "picam.db"

    def __init__(self) -> None:
        self._conn: sqlite3.Connection = sqlite3.connect(self.DB_FILE, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")          # faster concurrent access
        self._ensure_schema()

    # ---------- internal helpers ----------
    def _ensure_schema(self) -> None:
        cur = self._conn.cursor()
        cur.executescript("""
        CREATE TABLE IF NOT EXISTS Detections (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            object_id   INTEGER NOT NULL,
            object_type TEXT NOT NULL,   -- Face / Person
            zone        TEXT
        );

        CREATE TABLE IF NOT EXISTS Alerts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            alert_type  TEXT NOT NULL,   -- Idle / Unattended / RapidMotion / ...
            severity    TEXT NOT NULL    -- Low / Medium / High
        );

        CREATE TABLE IF NOT EXISTS DailyStats (
            date            TEXT PRIMARY KEY,
            total_faces     INTEGER DEFAULT 0,
            total_persons   INTEGER DEFAULT 0,
            total_motion_events INTEGER DEFAULT 0
        );
        """)
        self._conn.commit()

    # ---------- public insert API ----------
    def log_detection(self, obj_id: int, obj_type: str, zone: Optional[str] = None) -> None:
        sql = "INSERT INTO Detections (timestamp, object_id, object_type, zone) VALUES (?,?,?,?)"
        self._conn.execute(sql, (datetime.datetime.utcnow().isoformat(), obj_id, obj_type, zone))
        self._conn.commit()

    def log_alert(self, alert_type: str, severity: str = "Medium") -> None:
        sql = "INSERT INTO Alerts (timestamp, alert_type, severity) VALUES (?,?,?)"
        self._conn.execute(sql, (datetime.datetime.utcnow().isoformat(), alert_type, severity))
        self._conn.commit()

    def bump_daily_stat(self, faces: int = 0, persons: int = 0, motion_events: int = 0) -> None:
        """Increment counters for today (thread-safe via UPSERT)."""
        today = datetime.date.today().isoformat()
        cur = self._conn.cursor()
        cur.execute("""
        INSERT INTO DailyStats(date, total_faces, total_persons, total_motion_events)
        VALUES (?,?,?,?)
        ON CONFLICT(date) DO UPDATE SET
            total_faces = total_faces + ?,
            total_persons = total_persons + ?,
            total_motion_events = total_motion_events + ?
        """, (today, faces, persons, motion_events, faces, persons, motion_events))
        self._conn.commit()

    # ---------- read-only helpers (optional) ----------
    def get_todays_stats(self) -> Dict[str, int]:
        row = self._conn.execute(
            "SELECT total_faces, total_persons, total_motion_events FROM DailyStats WHERE date=?",
            (datetime.date.today().isoformat(),)
        ).fetchone()
        if not row:
            return {"faces": 0, "persons": 0, "motion_events": 0}
        return {"faces": row[0], "persons": row[1], "motion_events": row[2]}

    def close(self) -> None:
        self._conn.close()


# ------------------------------------------------------
# 2.  ZONE  (unchanged – only two tiny logging hooks added)
# ------------------------------------------------------
class Zone:
    """Define a zone (region of interest) with custom logic."""
    def __init__(self, name: str, points: List[Tuple[int, int]],
                 zone_type: str = "counter", idle_threshold: float = 120.0,
                 unattended_threshold: float = 300.0, motion_threshold: float = 0.01,
                 db: Optional[DatabaseManager] = None):
        self.name = name
        self.points = np.array(points, dtype=np.int32)
        self.zone_type = zone_type
        self.idle_threshold = idle_threshold
        self.unattended_threshold = unattended_threshold
        self.motion_threshold = motion_threshold
        self.db = db                       # <-- NEW: database reference

        # Calculate zone properties
        self.bbox = self.calculate_bounding_box()
        self.area = self.calculate_area()
        self.center = self.calculate_center()

        # Zone state
        self.persons_in_zone = set()
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

    # ---------- geometry helpers ----------
    def calculate_bounding_box(self) -> Tuple[int, int, int, int]:
        x_coords = [p[0] for p in self.points]
        y_coords = [p[1] for p in self.points]
        return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))

    def calculate_area(self) -> float:
        x = self.points[:, 0]; y = self.points[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def calculate_center(self) -> Tuple[int, int]:
        x_coords = [p[0] for p in self.points]
        y_coords = [p[1] for p in self.points]
        return ((min(x_coords) + max(x_coords)) // 2, (min(y_coords) + max(y_coords)) // 2)

    # ---------- hit-test ----------
    def is_point_in_zone(self, point: Tuple[float, float]) -> bool:
        x, y = point; n = len(self.points); inside = False
        p1x, p1y = self.points[0]
        for i in range(n + 1):
            p2x, p2y = self.points[i % n]
            if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
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
        return any(self.is_point_in_zone(c) for c in corners)

    # ---------- person life-cycle ----------
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
            print(f"[Zone '{self.name}'] Person {person_id} entered")
            if self.db:
                self.db.log_detection(obj_id=person_id, obj_type="Person", zone=self.name)
        else:
            data = self.person_data[person_id]
            data['last_update_time'] = current_time
            data['bbox'] = bbox
            if motion_score > 0:
                data['last_motion_time'] = current_time
                data['motion_history'].append(motion_score)
                data['total_motion'] += motion_score
                data['motion_samples'] += 1

    def remove_person(self, person_id: int, current_time: float = None):
        if person_id in self.person_data:
            if current_time is None:
                current_time = time.time()
            entry_time = self.person_data[person_id]['entry_time']
            self.total_occupancy_time += current_time - entry_time
            print(f"[Zone '{self.name}'] Person {person_id} left")
            del self.person_data[person_id]

    # ---------- idle / unattended logic ----------
    def check_idle_persons(self, current_time: float = None) -> List[int]:
        if current_time is None:
            current_time = time.time()
        idle_persons = []
        for pid, data in self.person_data.items():
            time_since_motion = current_time - data['last_motion_time']
            avg_motion = (data['total_motion'] / data['motion_samples']) if data['motion_samples'] else 0
            if time_since_motion > self.idle_threshold and avg_motion < self.motion_threshold:
                idle_persons.append(pid)
        return idle_persons

    def check_zone_status(self, current_time: float = None) -> Dict[str, Any]:
        if current_time is None:
            current_time = time.time()

        was_occupied = self.zone_occupied
        self.zone_occupied = bool(self.person_data)
        if self.zone_occupied:
            self.last_occupied_time = current_time
        else:
            self.last_vacant_time = current_time

        idle_persons = self.check_idle_persons(current_time)
        unattended_time = 0.0
        if not self.zone_occupied and self.last_occupied_time:
            unattended_time = current_time - self.last_occupied_time

        status_changed = False

        # Idle alert
        if idle_persons and not self.idle_alert_active:
            self.idle_alert_active = True
            self.idle_start_time = current_time
            status_changed = True
            print(f"[Zone '{self.name}'] IDLE ALERT: Persons {idle_persons}")
            if self.db:
                self.db.log_alert(f"Idle in {self.name}", "High")
        elif not idle_persons and self.idle_alert_active:
            self.idle_alert_active = False
            self.idle_start_time = None
            status_changed = True
            print(f"[Zone '{self.name}'] Idle alert cleared")

        # Unattended alert
        if unattended_time > self.unattended_threshold and not self.unattended_alert_active:
            self.unattended_alert_active = True
            self.unattended_start_time = current_time
            status_changed = True
            print(f"[Zone '{self.name}'] UNATTENDED ALERT")
            if self.db:
                self.db.log_alert(f"Unattended {self.name}", "Medium")
        elif unattended_time <= self.unattended_threshold and self.unattended_alert_active:
            self.unattended_alert_active = False
            self.unattended_start_time = None
            status_changed = True
            print(f"[Zone '{self.name}'] Unattended alert cleared")

        return {
            'zone_name': self.name,
            'zone_type': self.zone_type,
            'occupied': self.zone_occupied,
            'person_count': len(self.person_data),
            'person_ids': list(self.person_data.keys()),
            'idle_persons': idle_persons,
            'idle_alert': self.idle_alert_active,
            'idle_duration': current_time - self.idle_start_time if self.idle_start_time else 0,
            'unattended_time': unattended_time if not self.zone_occupied else 0,
            'unattended_alert': self.unattended_alert_active,
            'unattended_duration': current_time - self.unattended_start_time if self.unattended_start_time else 0,
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
            'avg_occupancy_time': self.total_occupancy_time / self.entry_count if self.entry_count else 0,
            'currently_occupied': self.zone_occupied,
            'current_persons': len(self.person_data),
            'idle_alert': self.idle_alert_active,
            'unattended_alert': self.unattended_alert_active
        }


# ------------------------------------------------------
# 3.  MotionDetector  (unchanged)
# ------------------------------------------------------
class MotionDetector:
    def __init__(self, history: int = 500, var_threshold: int = 16,
                 detect_shadows: bool = True, high_motion_threshold: float = 0.05,
                 min_motion_area: int = 100):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history, var_threshold, detect_shadows)
        self.high_motion_threshold = high_motion_threshold
        self.min_motion_area = min_motion_area
        self.motion_history: List[float] = []
        self.max_history = 30
        self.frame_width = 0
        self.frame_height = 0
        self.total_pixels = 0
        self.visualize = True

    def set_frame_size(self, width: int, height: int) -> None:
        self.frame_width, self.frame_height = width, height
        self.total_pixels = width * height

    def calculate_motion_score(self, frame: np.ndarray) -> Dict[str, Any]:
        fg_mask = self.bg_subtractor.apply(frame)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_areas, motion_centroids, valid_contours = [], [], []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > self.min_motion_area:
                motion_areas.append(area)
                valid_contours.append(cnt)
                M = cv2.moments(cnt)
                if M["m00"]:
                    motion_centroids.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))

        total_motion_area = sum(motion_areas)
        motion_score = total_motion_area / self.total_pixels if self.total_pixels else 0.0
        rapid_motion = motion_score > self.high_motion_threshold
        self.motion_history.append(motion_score)
        if len(self.motion_history) > self.max_history:
            self.motion_history.pop(0)
        avg_motion = float(np.mean(self.motion_history)) if self.motion_history else 0.0

        return {
            'motion_score': float(motion_score),
            'motion_percentage': float(motion_score * 100),
            'motion_areas': motion_areas,
            'motion_centroids': motion_centroids,
            'rapid_motion': rapid_motion,
            'avg_motion': avg_motion,
            'motion_mask': fg_mask,
            'contours': valid_contours,
            'total_motion_area': int(total_motion_area),
            'num_motion_regions': len(valid_contours)
        }


# ------------------------------------------------------
# 4.  PersonDetector  (unchanged)
# ------------------------------------------------------
class PersonDetector:
    def __init__(self, min_person_area: int = 1000, max_person_area: int = 50000):
        self.min_person_area = min_person_area
        self.max_person_area = max_person_area

    def detect_from_motion(self, motion_data: Dict, frame_shape: Tuple[int, int]) -> List[Dict]:
        persons = []
        for cnt in motion_data['contours']:
            area = cv2.contourArea(cnt)
            if self.min_person_area < area < self.max_person_area:
                x, y, w, h = cv2.boundingRect(cnt)
                persons.append({
                    'bbox': (x, y, x + w, y + h),
                    'centroid': (x + w // 2, y + h // 2),
                    'area': area,
                    'confidence': min(1.0, area / self.max_person_area)
                })
        return persons


# ------------------------------------------------------
# 5.  VideoProcessor  (same as before – only receives db handle)
# ------------------------------------------------------
class VideoProcessor:
    def __init__(self, source: Union[int, str] = 0,
                 confidence_threshold: float = 0.5,
                 enable_zones: bool = True,
                 enable_motion: bool = True,
                 enable_person_detection: bool = True,
                 motion_threshold: float = 0.05,
                 db: Optional[DatabaseManager] = None):
        self.source = source
        self.confidence_threshold = confidence_threshold
        self.running = False
        self.frame_count = 0
        self.fps = 0
        self.last_time = 0
        self.enable_zones = enable_zones
        self.enable_motion = enable_motion
        self.enable_person_detection = enable_person_detection
        self.db = db or DatabaseManager()  # create default if none supplied

        # video
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video source: {source}")
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps_video = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30

        # motion
        if enable_motion:
            self.motion_detector = MotionDetector(high_motion_threshold=motion_threshold)
            self.motion_detector.set_frame_size(self.width, self.height)
            print("✓ Motion detection initialized")

        # person
        if enable_person_detection:
            self.person_detector = PersonDetector()
            print("✓ Person detection initialized")

        # zones
        self.zones: List[Zone] = []
        self.zone_config_file = "zones_config.json"
        if enable_zones:
            self.load_zones_from_file()
            print(f"✓ Zone system initialized: {len(self.zones)} zones")

        # history
        self.motion_metrics_history: List[Dict] = []
        self.zone_status_history: List[Dict] = []
        self.max_history = 100

        print(f"✓ Video source: {source}")
        print(f"✓ Resolution: {self.width}x{self.height}")
        print(f"✓ Zones: {'Enabled' if enable_zones else 'Disabled'}")
        print(f"✓ Motion Analysis: {'Enabled' if enable_motion else 'Disabled'}")
        print(f"✓ Person Detection: {'Enabled' if enable_person_detection else 'Disabled'}")
        print(f"✓ SQLite logging: {self.db.DB_FILE}")

    # ---------- zones IO ----------
    def create_default_zones(self) -> None:
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
            (50, 100), (250, 100), (250, 300), (50, 300)
        ]
        self.zones = [
            Zone("Counter", counter_points, "counter", 120, 300, db=self.db),
            Zone("Table", table_points, "table", 180, 600, db=self.db),
            Zone("Entrance", entrance_points, "entrance", 60, 0, db=self.db)
        ]
        self.save_zones_to_file()

    def save_zones_to_file(self) -> None:
        zones_data = []
        for z in self.zones:
            zones_data.append({
                'name': z.name, 'points': z.points.tolist(), 'zone_type': z.zone_type,
                'idle_threshold': z.idle_threshold, 'unattended_threshold': z.unattended_threshold,
                'motion_threshold': z.motion_threshold
            })
        with open(self.zone_config_file, 'w') as f:
            json.dump(zones_data, f, indent=2)
        print(f"✓ Zones saved to {self.zone_config_file}")

    def load_zones_from_file(self) -> None:
        if not os.path.exists(self.zone_config_file):
            print("⚠ No zones config file found. Creating defaults...")
            return self.create_default_zones()
        try:
            with open(self.zone_config_file) as f:
                zones_data = json.load(f)
            self.zones = []
            for zd in zones_data:
                z = Zone(zd['name'], zd['points'], zd['zone_type'],
                         zd.get('idle_threshold', 120),
                         zd.get('unattended_threshold', 300),
                         zd.get('motion_threshold', 0.01),
                         db=self.db)
                self.zones.append(z)
            print(f"✓ Loaded {len(self.zones)} zones")
        except Exception as e:
            print(f"⚠ Error loading zones: {e}")
            self.create_default_zones()

    # ---------- drawing ----------
    def draw_zones(self, frame: np.ndarray) -> np.ndarray:
        for z in self.zones:
            color = (0, 165, 255)
            if z.zone_type == "counter":
                color = (255, 0, 255)
            elif z.zone_type == "table":
                color = (0, 255, 255)
            elif z.zone_type == "entrance":
                color = (0, 255, 0)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [z.points], color)
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
            cv2.polylines(frame, [z.points], True, color, 2)
            cv2.putText(frame, z.name, (z.center[0] - 30, z.center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Persons: {len(z.person_data)}", (z.center[0] - 30, z.center[1] + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            if z.idle_alert_active:
                cv2.putText(frame, "IDLE!", (z.bbox[0], z.bbox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if z.unattended_alert_active:
                cv2.putText(frame, "UNATTENDED!", (z.bbox[0], z.bbox[1] - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame

    def draw_persons(self, frame: np.ndarray, persons: List[Dict]) -> np.ndarray:
        for idx, p in enumerate(persons):
            x1, y1, x2, y2 = p['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"P{idx+1}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cx, cy = map(int, p['centroid'])
            cv2.circle(frame, (cx, cy), 3, (255, 0, 0), -1)
        return frame

    def draw_ui(self, frame: np.ndarray, motion_metrics: Dict, zone_statuses: List[Dict]) -> np.ndarray:
        fps_text = f'FPS: {self.fps:.1f}'
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        frame_text = f'Frame: {self.frame_count}'
        cv2.putText(frame, frame_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if motion_metrics:
            motion_text = f'Motion: {motion_metrics["motion_percentage"]:.1f}%'
            cv2.putText(frame, motion_text, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_pos = 120
        idle_alerts = sum(zs['idle_alert'] for zs in zone_statuses)
        unattended_alerts = sum(zs['unattended_alert'] for zs in zone_statuses)
        if idle_alerts:
            cv2.putText(frame, f'Idle Alerts: {idle_alerts}', (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y_pos += 25
        if unattended_alerts:
            cv2.putText(frame, f'Unattended Alerts: {unattended_alerts}', (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y_pos += 25

        # legend
        legend_y = self.height - 150
        cv2.putText(frame, 'Zone Legend:', (10, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, 'Counter (Purple)', (10, legend_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        cv2.putText(frame, 'Table (Yellow)', (10, legend_y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, 'Entrance (Green)', (10, legend_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return frame

    # ---------- zone processing ----------
    def process_zones(self, persons: List[Dict], motion_score: float = 0.0) -> List[Dict]:
        if not self.enable_zones:
            return []
        current_time = time.time()
        zone_statuses = []

        # timeout persons not updated for >1 s
        for z in self.zones:
            to_remove = [pid for pid, data in z.person_data.items()
                         if current_time - data['last_update_time'] > 1.0]
            for pid in to_remove:
                z.remove_person(pid, current_time)

        # update zones with current persons
        for idx, p in enumerate(persons):
            for z in self.zones:
                if z.is_bbox_in_zone(p['bbox']):
                    z.update_person(idx + 1, p['bbox'], motion_score, current_time)

        # collect statuses
        for z in self.zones:
            status = z.check_zone_status(current_time)
            zone_statuses.append(status)
            self.zone_status_history.append({'timestamp': current_time, 'zone_name': z.name, 'status': status})
            if len(self.zone_status_history) > self.max_history:
                self.zone_status_history.pop(0)
        return zone_statuses

    # ---------- fps ----------
    def calculate_fps(self) -> None:
        now = time.time()
        if self.last_time:
            self.fps = 1 / (now - self.last_time)
        self.last_time = now

    # ---------- frame pipeline ----------
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict, List[Dict]]:
        motion_metrics = {}
        persons = []
        zone_statuses = []

        if self.enable_motion:
            motion_metrics = self.motion_detector.calculate_motion_score(frame)
            if motion_metrics['num_motion_regions']:
                self.db.bump_daily_stat(motion_events=1)  # <-- log motion event

        if self.enable_person_detection and motion_metrics:
            persons = self.person_detector.detect_from_motion(motion_metrics, frame.shape[:2])

        if self.enable_zones:
            motion_score = motion_metrics.get('motion_score', 0.0) if motion_metrics else 0.0
            zone_statuses = self.process_zones(persons, motion_score)

        if self.enable_zones:
            frame = self.draw_zones(frame)
        if self.enable_person_detection:
            frame = self.draw_persons(frame, persons)

        self.calculate_fps()
        frame = self.draw_ui(frame, motion_metrics, zone_statuses)

        # console log every 30 frames
        if self.frame_count % 30 == 0:
            for zs in zone_statuses:
                if zs['idle_alert']:
                    print(f"[Frame {self.frame_count}] ZONE '{zs['zone_name']}': Idle alert {zs['idle_duration']:.1f}s")
                if zs['unattended_alert']:
                    print(f"[Frame {self.frame_count}] ZONE '{zs['zone_name']}': Unattended {zs['unattended_duration']:.1f}s")
        self.frame_count += 1
        return frame, motion_metrics, zone_statuses

    # ---------- statistics ----------
    def print_zone_statistics(self) -> None:
        print("\n" + "="*50)
        print("ZONE STATISTICS")
        print("="*50)
        for z in self.zones:
            st = z.get_statistics()
            print(f"\nZone: {st['name']} ({st['type']})")
            print(f"  Currently occupied: {st['currently_occupied']}")
            print(f"  Current persons: {st['current_persons']}")
            print(f"  Total entries: {st['total_entries']}")
            print(f"  Total occupancy time: {st['total_occupancy_time']:.1f}s")
            print(f"  Avg occupancy time: {st['avg_occupancy_time']:.1f}s")
            print(f"  Idle alert active: {st['idle_alert']}")
            print(f"  Unattended alert active: {st['unattended_alert']}")
        print("="*50)
        # also print todays DB stats
        db_stats = self.db.get_todays_stats()
        print(f"\nTODAY (from DB) -> Faces: {db_stats['faces']}  Persons: {db_stats['persons']}  MotionEvents: {db_stats['motion_events']}")
        print("="*50)

    # ---------- zone editor ----------
    def zone_editor_mode(self) -> None:
        print("\nZONE EDITOR MODE – click to add points, 'c' to complete, 'r' remove, 'q' quit")
        zone_points = []
        temp_frame = None

        def mouse_cb(event, x, y, flags, param):
            nonlocal zone_points, temp_frame
            if event == cv2.EVENT_LBUTTONDOWN:
                zone_points.append((x, y))
                print(f"Point added: {x,y}")
                cv2.circle(temp_frame, (x, y), 5, (0, 255, 0), -1)
                if len(zone_points) > 1:
                    cv2.line(temp_frame, zone_points[-2], zone_points[-1], (0, 255, 0), 2)

        cv2.namedWindow('Zone Editor')
        cv2.setMouseCallback('Zone Editor', mouse_cb)
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                temp_frame = frame.copy()
                # draw existing zones
                for z in self.zones:
                    cv2.polylines(temp_frame, [z.points], True, (0, 165, 255), 2)
                    cv2.putText(temp_frame, z.name, z.center, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                # draw current polygon
                for i, pt in enumerate(zone_points):
                    cv2.circle(temp_frame, pt, 5, (0, 255, 0), -1)
                    cv2.putText(temp_frame, str(i+1), (pt[0]+5, pt[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    if i:
                        cv2.line(temp_frame, zone_points[i-1], pt, (0, 255, 0), 2)
                cv2.putText(temp_frame, "Click add pts. 'c' complete. 'r' remove. 'q' quit.", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(temp_frame, f"Points: {len(zone_points)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('Zone Editor', temp_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    if len(zone_points) >= 3:
                        name = input("Zone name: ").strip() or "New Zone"
                        ztype = input("Zone type (counter/table/entrance): ").strip() or "table"
                        new_zone = Zone(name, zone_points, ztype, 120, 300, db=self.db)
                        self.zones.append(new_zone)
                        self.save_zones_to_file()
                        print(f"✓ Zone '{name}' created")
                        zone_points = []
                    else:
                        print("Need ≥3 points")
                elif key == ord('r'):
                    if zone_points:
                        zone_points.pop()
                        print("Removed last point")
                elif key == ord('q'):
                    break
        finally:
            cv2.destroyWindow('Zone Editor')
            print("Zone editor closed")

    # ---------- main loop ----------
    def run(self) -> None:
        self.running = True
        print("\n" + "="*50)
        print("PICAM – Zone Logic & Unattended Stations  (Session-5 Data-Logging)")
        print("="*50)
        print("Features:")
        print("  ✓ Feature 3: Idle Staff detection")
        print("  ✓ Feature 4: Unattended Station detection")
        print("  ✓ Feature 5: Zone-based person counting")
        print("  ✓ NEW: SQLite logging (Detections, Alerts, DailyStats)")
        print("\nControls:")
        print("  'q' or ESC – Quit")
        print("  's' – Screenshot")
        print("  'p' – Pause/resume")
        print("  'z' – Toggle zones")
        print("  'e' – Zone editor")
        print("  'd' – Print statistics (console + DB)")
        print("  'm' – Toggle motion detection")
        print("  '1' – Reset zones to default")
        print("="*50 + "\n")

        paused = False
        window_name = 'Picam – Zone Monitoring (Session-5)'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        try:
            while self.running:
                if not paused:
                    ret, frame = self.cap.read()
                    if not ret:
                        if isinstance(self.source, str):
                            replay = input("End of video. Replay? (y/n): ").lower()
                            if replay == 'y':
                                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                                continue
                        break
                    processed, motion_metrics, zone_statuses = self.process_frame(frame)
                    cv2.imshow(window_name, processed)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                elif key == ord('s'):
                    fname = f"picam_zones_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(fname, processed)
                    print(f"✓ Screenshot: {fname}")
                elif key == ord('p'):
                    paused = not paused
                    print("⏸️ Paused" if paused else "▶️ Resumed")
                elif key == ord('z'):
                    self.enable_zones = not self.enable_zones
                    print(f"Zones {'ON' if self.enable_zones else 'OFF'}")
                elif key == ord('e'):
                    self.zone_editor_mode()
                elif key == ord('d'):
                    self.print_zone_statistics()
                elif key == ord('m'):
                    self.enable_motion = not self.enable_motion
                    print(f"Motion {'ON' if self.enable_motion else 'OFF'}")
                elif key == ord('1'):
                    self.create_default_zones()
                    print("✓ Zones reset to default")
        except KeyboardInterrupt:
            print("\n⏹️ Interrupted")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop()

    # ---------- cleanup ----------
    def stop(self) -> None:
        self.running = False
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        self.db.close()
        print(f"\n✅ Stopped – frames processed: {self.frame_count}")
        self.print_zone_statistics()


# ------------------------------------------------------
# 6.  MAIN
# ------------------------------------------------------
def main() -> None:
    print("="*50)
    print("PICAM – Zone Monitoring System  (Session-5 Data-Logging)")
    print("="*50)
    print("Required: opencv-python, numpy")
    print("SQLite DB: picam.db (created automatically)")
    print("="*50)
    try:
        db = DatabaseManager()  # create one shared handle
        processor = VideoProcessor(
            source=0,  # webcam
            confidence_threshold=0.5,
            enable_zones=True,
            enable_motion=True,
            enable_person_detection=True,
            motion_threshold=0.05,
            db=db
        )
        processor.run()
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Webcam connected?")
        print("2. Try camera index 1")
        print("3. Another app using camera?")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()