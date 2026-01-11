# picam_behavioral_tracker.py
"""
Enhanced Picam Video Processor with Behavioral Tracking
Focuses on human behavior patterns, not just motion detection.
"""

import cv2
import numpy as np
import time
import json
import os
from datetime import datetime, date
import csv
from collections import defaultdict, deque
from typing import Optional, List, Tuple, Dict, Any
import threading

# (Include all the Zone, MotionDetector, PersonDetector classes from previous versions)
# (Include CSVManager class from previous versions)
# (Include VideoProcessor base class)

class BehavioralVideoProcessor(VideoProcessor):
    """Enhanced Video Processor with Behavioral Intelligence."""
    
    def __init__(self, source: Union[int, str] = 0, **kwargs):
        super().__init__(source, **kwargs)
        
        # Behavioral tracking
        self.behavioral_data = {
            'engagement_attempts': 0,
            'meaningful_interactions': 0,  # Dwell > 60 seconds
            'brief_encounters': 0,  # Dwell < 30 seconds
            'missed_opportunities': 0,
            'attention_gaps': [],
            'focus_shifts': 0
        }
        
        # Engagement tracking per zone
        self.zone_engagement = defaultdict(lambda: {
            'total_time': 0,
            'interaction_count': 0,
            'last_engagement': None,
            'engagement_gap': 0
        })
        
        print("✓ Behavioral Intelligence Layer Activated")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict, List[Dict]]:
        """Process frame with behavioral analysis."""
        motion_metrics = {}
        persons = []
        zone_statuses = []
        
        # Original processing
        if self.enable_motion:
            motion_metrics = self.motion_detector.calculate_motion_score(frame)
            self.log_motion_event(motion_metrics)
        
        if self.enable_person_detection and motion_metrics:
            persons = self.person_detector.detect_from_motion(motion_metrics, frame.shape[:2])
        
        if self.enable_zones:
            motion_score = motion_metrics.get('motion_score', 0.0) if motion_metrics else 0.0
            zone_statuses = self.process_zones(persons, motion_score)
            
            # Behavioral analysis
            self._analyze_behavioral_patterns(persons, zone_statuses, motion_score)
        
        # Draw everything
        if self.enable_zones:
            frame = self.draw_zones(frame)
        
        if self.enable_person_detection:
            frame = self.draw_persons(frame, persons)
        
        # Draw behavioral insights
        frame = self.draw_behavioral_insights(frame)
        
        # Update CSV stats periodically
        if self.frame_count % 60 == 0 and self.enable_csv_logging:
            self.update_csv_stats()
            self._log_behavioral_data()
        
        self.calculate_fps()
        frame = self.draw_ui(frame, motion_metrics, zone_statuses)
        
        self.frame_count += 1
        return frame, motion_metrics, zone_statuses
    
    def _analyze_behavioral_patterns(self, persons: List[Dict], zone_statuses: List[Dict], motion_score: float):
        """Analyze behavioral patterns in real-time."""
        current_time = time.time()
        
        # Track engagement in zones
        for zone_status in zone_statuses:
            zone_name = zone_status['zone_name']
            person_count = zone_status['person_count']
            
            if person_count > 0:
                # Update zone engagement
                zone_data = self.zone_engagement[zone_name]
                
                if zone_data['last_engagement']:
                    gap = current_time - zone_data['last_engagement']
                    if gap > 30:  # 30 seconds gap
                        self.behavioral_data['attention_gaps'].append({
                            'zone': zone_name,
                            'gap_seconds': gap,
                            'timestamp': datetime.now().isoformat()
                        })
                        zone_data['engagement_gap'] = gap
                
                zone_data['last_engagement'] = current_time
                zone_data['total_time'] += 1  # Increment frame count
                
                # Check for meaningful interaction (counter zone specific)
                if zone_name == 'Counter' and person_count > 0:
                    self.behavioral_data['engagement_attempts'] += 1
                    
                    # Simulate interaction quality based on motion
                    if motion_score > 0.1:  # Some activity
                        self.behavioral_data['meaningful_interactions'] += 1
                    else:
                        self.behavioral_data['brief_encounters'] += 1
        
        # Track missed opportunities (people in entrance but not moving to counter)
        entrance_occupied = any(z['zone_name'] == 'Entrance' and z['person_count'] > 0 
                               for z in zone_statuses)
        counter_occupied = any(z['zone_name'] == 'Counter' and z['person_count'] > 0 
                              for z in zone_statuses)
        
        if entrance_occupied and not counter_occupied:
            self.behavioral_data['missed_opportunities'] += 1
    
    def _log_behavioral_data(self):
        """Log behavioral data to CSV."""
        if not self.enable_csv_logging:
            return
        
        timestamp = datetime.now()
        
        # Calculate engagement metrics
        total_attempts = self.behavioral_data['engagement_attempts']
        meaningful = self.behavioral_data['meaningful_interactions']
        brief = self.behavioral_data['brief_encounters']
        
        engagement_rate = meaningful / total_attempts if total_attempts > 0 else 0
        missed_rate = self.behavioral_data['missed_opportunities'] / max(1, total_attempts + brief)
        
        # Log to behavioral CSV
        behavioral_file = os.path.join(self.csv_manager.output_dir, 
                                      f"behavioral_{date.today().strftime('%Y%m%d')}.csv")
        
        # Create file if it doesn't exist
        if not os.path.exists(behavioral_file):
            with open(behavioral_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'engagement_attempts', 'meaningful_interactions',
                    'brief_encounters', 'engagement_rate', 'missed_opportunities',
                    'attention_gaps_count', 'focus_shifts'
                ])
        
        # Append data
        with open(behavioral_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp.isoformat(),
                total_attempts,
                meaningful,
                brief,
                engagement_rate,
                self.behavioral_data['missed_opportunities'],
                len(self.behavioral_data['attention_gaps']),
                self.behavioral_data['focus_shifts']
            ])
    
    def draw_behavioral_insights(self, frame: np.ndarray) -> np.ndarray:
        """Draw behavioral insights on frame."""
        height, width = frame.shape[:2]
        
        # Draw engagement metrics
        metrics_y = height - 150
        
        # Engagement rate
        total = self.behavioral_data['engagement_attempts']
        meaningful = self.behavioral_data['meaningful_interactions']
        
        if total > 0:
            rate = meaningful / total * 100
            color = (0, 255, 0) if rate > 50 else (0, 165, 255) if rate > 25 else (0, 0, 255)
            
            cv2.putText(frame, f"Engagement: {rate:.1f}%", 
                       (width - 250, metrics_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Missed opportunities
        missed = self.behavioral_data['missed_opportunities']
        if missed > 0:
            cv2.putText(frame, f"Missed: {missed}", 
                       (width - 250, metrics_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Zone engagement status
        zone_y = metrics_y + 60
        for zone_name, zone_data in self.zone_engagement.items():
            if zone_data['total_time'] > 0:
                gap = zone_data['engagement_gap']
                status = "ACTIVE" if gap < 10 else "IDLE" if gap < 30 else "INACTIVE"
                color = (0, 255, 0) if status == "ACTIVE" else (0, 165, 255) if status == "IDLE" else (0, 0, 255)
                
                cv2.putText(frame, f"{zone_name}: {status}", 
                           (width - 250, zone_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                zone_y += 20
        
        return frame


def main():
    """Run the Behavioral Video Processor."""
    print("="*80)
    print("PICAM BEHAVIORAL TRACKER")
    print("="*80)
    print("Tracking human behavior patterns for truth intelligence.")
    print("\nFocus: Engagement, Attention, Value Capture")
    print("="*80)
    
    try:
        processor = BehavioralVideoProcessor(
            source=0,  # Webcam
            confidence_threshold=0.5,
            enable_zones=True,
            enable_motion=True,
            enable_person_detection=True,
            enable_csv_logging=True,
            motion_threshold=0.05
        )
        
        processor.run()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()