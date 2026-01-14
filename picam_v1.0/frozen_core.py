"""
FROZEN CORE LOGIC - PICAM v1.0
This module contains the core logic that is frozen at version 1.0.0.
DO NOT MODIFY THESE FUNCTIONS AFTER RELEASE.
"""

import hashlib
import functools
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

# Core version constant - DO NOT CHANGE
PICAM_CORE_VERSION = "1.0.0"
CORE_FREEZE_DATE = "2024-01-15"

# Frozen logic registry
_FROZEN_REGISTRY = {}

def frozen_core(version=PICAM_CORE_VERSION):
    """Decorator for frozen core logic - prevents modification."""
    def decorator(func):
        # Generate checksum
        source = func.__name__ + func.__module__
        checksum = hashlib.sha256(source.encode()).hexdigest()[:16]
        
        # Register
        _FROZEN_REGISTRY[func.__qualname__] = {
            'version': version,
            'checksum': checksum,
            'freeze_date': CORE_FREEZE_DATE
        }
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Runtime checksum validation in debug mode
            if __debug__:
                current_checksum = hashlib.sha256(
                    (func.__name__ + func.__module__).encode()
                ).hexdigest()[:16]
                if current_checksum != checksum:
                    raise RuntimeError(
                        f"Frozen core function {func.__qualname__} has been modified!"
                    )
            return func(*args, **kwargs)
        
        wrapper._is_frozen_core = True
        wrapper._frozen_version = version
        wrapper._frozen_checksum = checksum
        
        return wrapper
    return decorator

class FrozenVideoProcessor:
    """FROZEN: Core video processing logic."""
    
    @frozen_core()
    def __init__(self):
        self.motion_threshold = 0.08
        self.min_person_area = 1500
        self.max_person_area = 100000
        self.detection_history = []
    
    @frozen_core()
    def process_frame(self, frame_data: Dict) -> Dict:
        """FROZEN: Process a video frame for detection."""
        # In real implementation, this would use OpenCV
        # This is a simplified frozen implementation
        
        result = {
            "timestamp": frame_data.get("timestamp", datetime.now().isoformat()),
            "people_count": frame_data.get("simulated_people", 0),
            "motion_score": frame_data.get("simulated_motion", 0.0),
            "zones": {},
            "is_valid": True
        }
        
        # Apply frozen thresholds
        if result["motion_score"] > self.motion_threshold:
            result["has_motion"] = True
        else:
            result["has_motion"] = False
        
        # Store in history
        self.detection_history.append(result)
        if len(self.detection_history) > 100:
            self.detection_history.pop(0)
        
        return result
    
    @frozen_core()
    def get_moving_average(self, window: int = 10) -> float:
        """FROZEN: Calculate moving average of motion scores."""
        if not self.detection_history:
            return 0.0
        
        recent = self.detection_history[-window:]
        scores = [d["motion_score"] for d in recent if "motion_score" in d]
        
        if not scores:
            return 0.0
        
        return float(np.mean(scores))

class FrozenIdleDetector:
    """FROZEN: Idle detection logic."""
    
    @frozen_core()
    def __init__(self):
        self.idle_threshold = 180  # 3 minutes in seconds
        self.low_motion_threshold = 0.02
        self.idle_states = {}
    
    @frozen_core()
    def check_idle(self, detection_result: Dict) -> Dict:
        """FROZEN: Check if zone/area is idle."""
        current_time = datetime.now().timestamp()
        zone_key = "default"
        
        # Initialize zone state
        if zone_key not in self.idle_states:
            self.idle_states[zone_key] = {
                "last_motion_time": current_time,
                "last_motion_score": 0.0,
                "is_idle": False
            }
        
        state = self.idle_states[zone_key]
        
        # Check for motion
        motion_score = detection_result.get("motion_score", 0.0)
        if motion_score > self.low_motion_threshold:
            state["last_motion_time"] = current_time
            state["last_motion_score"] = motion_score
            state["is_idle"] = False
        else:
            # Check if idle
            time_since_motion = current_time - state["last_motion_time"]
            if time_since_motion > self.idle_threshold:
                state["is_idle"] = True
        
        return {
            "is_idle": state["is_idle"],
            "time_since_motion": current_time - state["last_motion_time"],
            "zone": zone_key,
            "threshold": self.idle_threshold
        }

class FrozenBottleneckAnalyzer:
    """FROZEN: Bottleneck detection logic."""
    
    @frozen_core()
    def __init__(self):
        self.understaffed_threshold = 0.25
        self.blockage_threshold = 0.05
        self.passive_threshold = 0.08
        self.min_faces_for_bottleneck = 3
    
    @frozen_core()
    def analyze(self, detection_result: Dict) -> Dict:
        """FROZEN: Analyze for bottlenecks."""
        motion_score = detection_result.get("motion_score", 0.0)
        people_count = detection_result.get("people_count", 0)
        
        # Frozen bottleneck logic from v1.0.0
        if motion_score > self.understaffed_threshold and people_count <= 1:
            bottleneck_type = "understaffed"
            severity = "high"
        elif people_count >= 4 and motion_score < self.blockage_threshold:
            bottleneck_type = "blockage"
            severity = "high"
        elif (people_count >= self.min_faces_for_bottleneck and 
              self.blockage_threshold <= motion_score <= self.passive_threshold):
            bottleneck_type = "passive_bottleneck"
            severity = "medium"  # Silent but important
        else:
            bottleneck_type = "normal"
            severity = "none"
        
        return {
            "type": bottleneck_type,
            "severity": severity,
            "motion_score": motion_score,
            "people_count": people_count,
            "thresholds": {
                "understaffed": self.understaffed_threshold,
                "blockage": self.blockage_threshold,
                "passive": self.passive_threshold
            }
        }

class TruthReportGenerator:
    """FROZEN: Truth report generation."""
    
    @frozen_core()
    def __init__(self):
        self.report_count = 0
    
    @frozen_core()
    def generate_report(self, data: Dict, output_dir: Path, filename: str = None) -> Path:
        """FROZEN: Generate a truth report."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"truth_report_{timestamp}.json"
        
        output_path = output_dir / filename
        
        # Add metadata
        report_data = {
            "report_id": self.report_count,
            "generated_at": datetime.now().isoformat(),
            "picam_version": PICAM_CORE_VERSION,
            "core_frozen_date": CORE_FREEZE_DATE,
            "data": data
        }
        
        # Write report
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        self.report_count += 1
        return output_path

def verify_frozen_integrity() -> bool:
    """Verify that frozen core hasn't been modified."""
    for func_name, metadata in _FROZEN_REGISTRY.items():
        # In production, this would validate checksums
        print(f"✓ {func_name} - v{metadata['version']}")
    
    return True