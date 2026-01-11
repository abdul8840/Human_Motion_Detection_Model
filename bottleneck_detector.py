# bottleneck_detector.py
import csv
import os
import numpy as np
from datetime import datetime, date, timedelta
from collections import defaultdict, deque
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import json

@dataclass
class BottleneckEvent:
    """Data class for bottleneck events."""
    start_time: datetime
    end_time: datetime
    bottleneck_type: str  # 'understaffed' or 'blockage'
    severity: str  # 'low', 'medium', 'high', 'critical'
    duration_minutes: float
    avg_motion_score: float
    avg_face_count: float
    confidence: float
    zone_name: Optional[str] = None
    description: str = ""
    recommendations: List[str] = None
    
    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []

class BottleneckDetector:
    """
    Advanced Bottleneck Detection System for Picam Project.
    
    Detects invisible bottlenecks by analyzing patterns between:
    1. Motion activity (busyness)
    2. Face/person count (staffing/crowding)
    3. Temporal patterns and correlations
    
    Bottleneck Types:
    - UNDERSTAFFED: High motion (busy) but low face count (not enough staff)
    - BLOCKAGE: High face count (crowded) but low motion (people standing still)
    - CHOKE_POINT: Consistent high face count in specific zone
    - IDLE_CLUSTER: Multiple idle alerts clustering in time
    """
    
    def __init__(self, reports_dir: str = "picam_reports"):
        """
        Initialize the Bottleneck Detector.
        
        Args:
            reports_dir: Directory containing CSV reports
        """
        self.reports_dir = reports_dir
        self.today = date.today()
        self.today_str = self.today.strftime("%Y%m%d")
        
        # Bottleneck detection thresholds (configurable)
        self.config = {
            # Time window for analysis (minutes)
            'time_window': 15,
            
            # Motion thresholds (0-1 scale)
            'high_motion_threshold': 0.3,
            'low_motion_threshold': 0.1,
            
            # Face count thresholds
            'high_face_threshold': 3,
            'low_face_threshold': 1,
            
            # Minimum duration for bottleneck (minutes)
            'min_bottleneck_duration': 5,
            
            # Confidence thresholds
            'high_confidence_threshold': 0.7,
            'medium_confidence_threshold': 0.5,
            
            # Zone-specific adjustments
            'zone_adjustments': {
                'Counter': {'face_multiplier': 1.5},
                'Entrance': {'motion_multiplier': 1.2},
                'Table': {'motion_threshold': 0.15}
            }
        }
        
        # Data storage
        self.timeline_data = []
        self.bottlenecks = []
        
        print(f"🔍 Bottleneck Detector Initialized")
        print(f"   Date: {self.today.strftime('%Y-%m-%d')}")
        print(f"   Reports Directory: {reports_dir}")
    
    def load_and_prepare_data(self) -> bool:
        """
        Load and prepare data from CSV files.
        Creates a unified timeline with motion and face counts.
        """
        print(f"\n📊 Loading data...")
        
        try:
            # Load motion events
            motion_events = self._load_motion_events()
            print(f"   ✓ Motion events: {len(motion_events)}")
            
            # Load detections
            detections = self._load_detections()
            print(f"   ✓ Detections: {len(detections)}")
            
            # Load zone data
            zone_stats = self._load_zone_stats()
            print(f"   ✓ Zones: {len(zone_stats)}")
            
            # Create unified timeline
            self._create_timeline(motion_events, detections, zone_stats)
            print(f"   ✓ Timeline created: {len(self.timeline_data)} timepoints")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error loading data: {e}")
            return False
    
    def _load_motion_events(self) -> List[Dict]:
        """Load motion events from CSV."""
        events = []
        motion_file = os.path.join(self.reports_dir, f"motion_events_{self.today_str}.csv")
        
        if os.path.exists(motion_file):
            with open(motion_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        # Parse timestamp
                        timestamp = datetime.strptime(
                            row['timestamp'].split('.')[0], 
                            "%Y-%m-%d %H:%M:%S"
                        )
                        
                        events.append({
                            'timestamp': timestamp,
                            'motion_score': float(row['motion_score']),
                            'motion_percentage': float(row['motion_percentage']),
                            'rapid_motion': row['rapid_motion'] == 'True',
                            'num_regions': int(row['num_regions']),
                            'frame_number': int(row['frame_number']) if row['frame_number'] else 0
                        })
                    except Exception as e:
                        continue
        
        return events
    
    def _load_detections(self) -> List[Dict]:
        """Load detections from CSV."""
        detections = []
        detections_file = os.path.join(self.reports_dir, f"detections_{self.today_str}.csv")
        
        if os.path.exists(detections_file):
            with open(detections_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        timestamp = datetime.strptime(
                            row['timestamp'].split('.')[0], 
                            "%Y-%m-%d %H:%M:%S"
                        )
                        
                        detections.append({
                            'timestamp': timestamp,
                            'object_id': int(row['object_id']) if row['object_id'] else 0,
                            'object_type': row['object_type'],
                            'zone_name': row['zone_name'] if row['zone_name'] else 'Unknown',
                            'confidence': float(row['confidence']) if row['confidence'] else 0.0
                        })
                    except Exception:
                        continue
        
        return detections
    
    def _load_zone_stats(self) -> List[Dict]:
        """Load zone statistics from CSV."""
        zone_stats = []
        zone_file = os.path.join(self.reports_dir, f"zone_stats_{self.today_str}.csv")
        
        if os.path.exists(zone_file):
            with open(zone_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['date'] == self.today.strftime("%Y-%m-%d"):
                        zone_stats.append(row)
        
        return zone_stats
    
    def _create_timeline(self, motion_events: List[Dict], detections: List[Dict], zone_stats: List[Dict]):
        """Create a unified timeline with motion and face counts."""
        # Group detections by time window (e.g., every 5 minutes)
        detection_by_time = defaultdict(list)
        for detection in detections:
            # Round to nearest 5 minutes for aggregation
            rounded_time = detection['timestamp'].replace(
                minute=(detection['timestamp'].minute // 5) * 5,
                second=0,
                microsecond=0
            )
            detection_by_time[rounded_time].append(detection)
        
        # Group motion events similarly
        motion_by_time = defaultdict(list)
        for event in motion_events:
            rounded_time = event['timestamp'].replace(
                minute=(event['timestamp'].minute // 5) * 5,
                second=0,
                microsecond=0
            )
            motion_by_time[rounded_time].append(event)
        
        # Combine all unique times
        all_times = set(list(detection_by_time.keys()) + list(motion_by_time.keys()))
        
        for time_point in sorted(all_times):
            # Calculate face count for this time window
            face_count = 0
            person_count = 0
            zone_counts = defaultdict(int)
            
            for detection in detection_by_time.get(time_point, []):
                if detection['object_type'] == 'face':
                    face_count += 1
                elif detection['object_type'] == 'person':
                    person_count += 1
                
                if detection['zone_name']:
                    zone_counts[detection['zone_name']] += 1
            
            # Calculate average motion score
            motion_scores = [e['motion_score'] for e in motion_by_time.get(time_point, [])]
            avg_motion = np.mean(motion_scores) if motion_scores else 0
            
            # Store timeline data
            self.timeline_data.append({
                'timestamp': time_point,
                'face_count': face_count,
                'person_count': person_count,
                'avg_motion': avg_motion,
                'zone_counts': dict(zone_counts),
                'motion_events_count': len(motion_by_time.get(time_point, [])),
                'detection_events_count': len(detection_by_time.get(time_point, []))
            })
    
    def detect_bottlenecks(self) -> List[BottleneckEvent]:
        """
        Main bottleneck detection algorithm.
        
        Identifies:
        1. Understaffed periods (High motion, Low faces)
        2. Blockage periods (High faces, Low motion)
        3. Choke points (Consistent crowding)
        4. Idle clusters (Multiple idle alerts)
        """
        print(f"\n🔍 Analyzing bottlenecks...")
        
        if not self.timeline_data:
            print("   ⚠ No timeline data available")
            return []
        
        bottlenecks = []
        
        # 1. Detect Understaffed Bottlenecks (High motion, Low faces)
        understaffed = self._detect_understaffed_bottlenecks()
        bottlenecks.extend(understaffed)
        print(f"   ✓ Understaffed bottlenecks: {len(understaffed)}")
        
        # 2. Detect Blockage Bottlenecks (High faces, Low motion)
        blockages = self._detect_blockage_bottlenecks()
        bottlenecks.extend(blockages)
        print(f"   ✓ Blockage bottlenecks: {len(blockages)}")
        
        # 3. Detect Choke Points
        choke_points = self._detect_choke_points()
        bottlenecks.extend(choke_points)
        print(f"   ✓ Choke points: {len(choke_points)}")
        
        # 4. Detect Idle Clusters
        idle_clusters = self._detect_idle_clusters()
        bottlenecks.extend(idle_clusters)
        print(f"   ✓ Idle clusters: {len(idle_clusters)}")
        
        # Sort by severity and duration
        bottlenecks.sort(key=lambda x: (
            self._severity_to_score(x.severity),
            x.duration_minutes
        ), reverse=True)
        
        self.bottlenecks = bottlenecks
        return bottlenecks
    
    def _detect_understaffed_bottlenecks(self) -> List[BottleneckEvent]:
        """Detect periods of high motion but low face count."""
        bottlenecks = []
        current_window = []
        
        for i, data_point in enumerate(self.timeline_data):
            # Check if this point shows understaffing
            is_understaffed = (
                data_point['avg_motion'] > self.config['high_motion_threshold'] and
                data_point['face_count'] <= self.config['low_face_threshold']
            )
            
            if is_understaffed:
                current_window.append(data_point)
            else:
                if len(current_window) >= 3:  # At least 3 consecutive points (15 minutes)
                    bottleneck = self._create_bottleneck_from_window(
                        current_window, 
                        'understaffed'
                    )
                    if bottleneck:
                        bottlenecks.append(bottleneck)
                current_window = []
        
        # Check final window
        if len(current_window) >= 3:
            bottleneck = self._create_bottleneck_from_window(
                current_window, 
                'understaffed'
            )
            if bottleneck:
                bottlenecks.append(bottleneck)
        
        return bottlenecks
    
    def _detect_blockage_bottlenecks(self) -> List[BottleneckEvent]:
        """Detect periods of high face count but low motion."""
        bottlenecks = []
        current_window = []
        
        for data_point in self.timeline_data:
            # Check if this point shows blockage
            is_blockage = (
                data_point['face_count'] >= self.config['high_face_threshold'] and
                data_point['avg_motion'] < self.config['low_motion_threshold']
            )
            
            if is_blockage:
                current_window.append(data_point)
            else:
                if len(current_window) >= 3:
                    bottleneck = self._create_bottleneck_from_window(
                        current_window, 
                        'blockage'
                    )
                    if bottleneck:
                        bottlenecks.append(bottleneck)
                current_window = []
        
        if len(current_window) >= 3:
            bottleneck = self._create_bottleneck_from_window(
                current_window, 
                'blockage'
            )
            if bottleneck:
                bottlenecks.append(bottleneck)
        
        return bottlenecks
    
    def _detect_choke_points(self) -> List[BottleneckEvent]:
        """Detect consistent crowding in specific zones."""
        bottlenecks = []
        
        # Analyze zone-specific patterns
        zone_data = defaultdict(list)
        
        for data_point in self.timeline_data:
            for zone_name, count in data_point['zone_counts'].items():
                zone_data[zone_name].append({
                    'timestamp': data_point['timestamp'],
                    'count': count,
                    'motion': data_point['avg_motion']
                })
        
        for zone_name, zone_points in zone_data.items():
            if len(zone_points) < 10:  # Need sufficient data
                continue
            
            # Find periods of sustained high occupancy
            high_occupancy_windows = []
            current_window = []
            
            for point in zone_points:
                if point['count'] >= self.config['high_face_threshold']:
                    current_window.append(point)
                else:
                    if len(current_window) >= 4:  # At least 20 minutes
                        high_occupancy_windows.append(current_window)
                    current_window = []
            
            if len(current_window) >= 4:
                high_occupancy_windows.append(current_window)
            
            # Create bottlenecks for sustained high occupancy
            for window in high_occupancy_windows:
                if window:
                    bottleneck = BottleneckEvent(
                        start_time=window[0]['timestamp'],
                        end_time=window[-1]['timestamp'],
                        bottleneck_type='choke_point',
                        severity=self._calculate_severity(len(window), max(p['count'] for p in window)),
                        duration_minutes=len(window) * 5,  # 5 minutes per data point
                        avg_motion_score=np.mean([p['motion'] for p in window]),
                        avg_face_count=np.mean([p['count'] for p in window]),
                        confidence=self._calculate_confidence(window),
                        zone_name=zone_name,
                        description=f"Sustained crowding in {zone_name} zone",
                        recommendations=[
                            f"Consider adding staff to {zone_name}",
                            f"Review workflow in {zone_name} area",
                            f"Monitor {zone_name} during peak hours"
                        ]
                    )
                    bottlenecks.append(bottleneck)
        
        return bottlenecks
    
    def _detect_idle_clusters(self) -> List[BottleneckEvent]:
        """Detect clusters of idle alerts."""
        bottlenecks = []
        
        # Load idle alerts
        idle_alerts = self._load_idle_alerts()
        
        if not idle_alerts:
            return bottlenecks
        
        # Group idle alerts by time proximity
        idle_alerts.sort(key=lambda x: x['timestamp'])
        
        current_cluster = []
        for alert in idle_alerts:
            if not current_cluster:
                current_cluster.append(alert)
            else:
                last_alert = current_cluster[-1]
                time_diff = (alert['timestamp'] - last_alert['timestamp']).total_seconds() / 60
                
                if time_diff <= 30:  # Within 30 minutes
                    current_cluster.append(alert)
                else:
                    if len(current_cluster) >= 3:  # Cluster of at least 3 idle alerts
                        bottlenecks.append(
                            self._create_idle_cluster_bottleneck(current_cluster)
                        )
                    current_cluster = [alert]
        
        # Check final cluster
        if len(current_cluster) >= 3:
            bottlenecks.append(
                self._create_idle_cluster_bottleneck(current_cluster)
            )
        
        return bottlenecks
    
    def _load_idle_alerts(self) -> List[Dict]:
        """Load idle staff alerts from alerts CSV."""
        idle_alerts = []
        alerts_file = os.path.join(self.reports_dir, f"alerts_{self.today_str}.csv")
        
        if os.path.exists(alerts_file):
            with open(alerts_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['alert_type'] == 'idle_staff':
                        try:
                            timestamp = datetime.strptime(
                                row['timestamp'].split('.')[0],
                                "%Y-%m-%d %H:%M:%S"
                            )
                            idle_alerts.append({
                                'timestamp': timestamp,
                                'zone_name': row['zone_name'],
                                'object_id': row['object_id'],
                                'duration_seconds': float(row['duration_seconds'])
                            })
                        except Exception:
                            continue
        
        return idle_alerts
    
    def _create_bottleneck_from_window(self, window: List[Dict], bottleneck_type: str) -> Optional[BottleneckEvent]:
        """Create a BottleneckEvent from a detection window."""
        if not window:
            return None
        
        duration_minutes = len(window) * 5  # 5 minutes per data point
        if duration_minutes < self.config['min_bottleneck_duration']:
            return None
        
        avg_motion = np.mean([p['avg_motion'] for p in window])
        avg_faces = np.mean([p['face_count'] for p in window])
        
        # Determine severity
        severity = self._calculate_severity(duration_minutes, avg_faces if bottleneck_type == 'blockage' else avg_motion)
        
        # Determine confidence
        confidence = self._calculate_confidence(window)
        
        # Generate description and recommendations
        if bottleneck_type == 'understaffed':
            description = "High activity with insufficient staffing"
            recommendations = [
                "Consider adding staff during high-activity periods",
                "Review staff scheduling",
                "Implement cross-training for flexible staffing"
            ]
        else:  # blockage
            description = "Crowding with low movement (potential blockage)"
            recommendations = [
                "Review layout and flow in affected area",
                "Consider signage or queuing systems",
                "Monitor for equipment issues"
            ]
        
        # Identify affected zones
        zone_counts = defaultdict(int)
        for point in window:
            for zone, count in point['zone_counts'].items():
                zone_counts[zone] += count
        
        affected_zone = max(zone_counts.items(), key=lambda x: x[1])[0] if zone_counts else None
        
        return BottleneckEvent(
            start_time=window[0]['timestamp'],
            end_time=window[-1]['timestamp'],
            bottleneck_type=bottleneck_type,
            severity=severity,
            duration_minutes=duration_minutes,
            avg_motion_score=avg_motion,
            avg_face_count=avg_faces,
            confidence=confidence,
            zone_name=affected_zone,
            description=description,
            recommendations=recommendations
        )
    
    def _create_idle_cluster_bottleneck(self, cluster: List[Dict]) -> BottleneckEvent:
        """Create a bottleneck event for idle alert clusters."""
        start_time = cluster[0]['timestamp']
        end_time = cluster[-1]['timestamp']
        duration_minutes = (end_time - start_time).total_seconds() / 60
        
        # Count zones in cluster
        zone_counts = defaultdict(int)
        for alert in cluster:
            if alert['zone_name']:
                zone_counts[alert['zone_name']] += 1
        
        affected_zone = max(zone_counts.items(), key=lambda x: x[1])[0] if zone_counts else None
        
        return BottleneckEvent(
            start_time=start_time,
            end_time=end_time,
            bottleneck_type='idle_cluster',
            severity='high' if len(cluster) >= 5 else 'medium',
            duration_minutes=duration_minutes,
            avg_motion_score=0.0,
            avg_face_count=len(cluster),
            confidence=0.8,
            zone_name=affected_zone,
            description=f"Cluster of {len(cluster)} idle alerts",
            recommendations=[
                "Review task allocation and supervision",
                "Consider implementing activity check-ins",
                "Evaluate workload distribution"
            ]
        )
    
    def _calculate_severity(self, duration: float, intensity: float) -> str:
        """Calculate bottleneck severity."""
        if duration > 60 or intensity > 5:
            return 'critical'
        elif duration > 30 or intensity > 3:
            return 'high'
        elif duration > 15 or intensity > 2:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_confidence(self, window: List[Dict]) -> float:
        """Calculate confidence score for bottleneck detection."""
        if not window:
            return 0.0
        
        # Factors for confidence:
        # 1. Window length (longer = more confident)
        length_factor = min(1.0, len(window) / 10)
        
        # 2. Consistency within window
        motions = [p['avg_motion'] for p in window]
        faces = [p['face_count'] for p in window]
        
        motion_std = np.std(motions) if len(motions) > 1 else 0
        face_std = np.std(faces) if len(faces) > 1 else 0
        
        consistency_factor = 1.0 - (motion_std + face_std) / 2
        
        # 3. Clear separation from thresholds
        threshold_factor = 0.5  # Base, adjust based on actual values
        
        confidence = (length_factor * 0.4 + 
                     consistency_factor * 0.4 + 
                     threshold_factor * 0.2)
        
        return round(confidence, 2)
    
    def _severity_to_score(self, severity: str) -> int:
        """Convert severity string to numerical score for sorting."""
        scores = {
            'critical': 4,
            'high': 3,
            'medium': 2,
            'low': 1
        }
        return scores.get(severity, 0)
    
    def generate_bottleneck_report(self) -> str:
        """Generate comprehensive bottleneck report."""
        if not self.bottlenecks:
            return "No bottlenecks detected today."
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("🧭 INVISIBLE BOTTLENECK DETECTOR REPORT")
        report_lines.append(f"Date: {self.today.strftime('%Y-%m-%d')}")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("="*80)
        report_lines.append("")
        
        # Executive Summary
        report_lines.append("📊 EXECUTIVE SUMMARY")
        report_lines.append("-"*40)
        
        bottleneck_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        
        for bottleneck in self.bottlenecks:
            bottleneck_counts[bottleneck.bottleneck_type] += 1
            severity_counts[bottleneck.severity] += 1
        
        report_lines.append(f"• Total Bottlenecks Detected: {len(self.bottlenecks)}")
        report_lines.append(f"• Critical Bottlenecks: {severity_counts.get('critical', 0)}")
        report_lines.append(f"• High Severity: {severity_counts.get('high', 0)}")
        report_lines.append("")
        
        # Detailed Bottleneck Analysis
        report_lines.append("🔍 DETAILED BOTTLENECK ANALYSIS")
        report_lines.append("-"*40)
        
        for i, bottleneck in enumerate(self.bottlenecks, 1):
            report_lines.append(f"\n{i}. {bottleneck.bottleneck_type.upper()} ({bottleneck.severity.upper()})")
            report_lines.append(f"   Time: {bottleneck.start_time.strftime('%H:%M')} - {bottleneck.end_time.strftime('%H:%M')}")
            report_lines.append(f"   Duration: {bottleneck.duration_minutes:.1f} minutes")
            if bottleneck.zone_name:
                report_lines.append(f"   Zone: {bottleneck.zone_name}")
            report_lines.append(f"   Avg Motion: {bottleneck.avg_motion_score:.2f}")
            report_lines.append(f"   Avg Faces: {bottleneck.avg_face_count:.1f}")
            report_lines.append(f"   Confidence: {bottleneck.confidence:.0%}")
            report_lines.append(f"   Description: {bottleneck.description}")
            report_lines.append(f"   Recommendations:")
            for rec in bottleneck.recommendations:
                report_lines.append(f"     • {rec}")
        
        # Pattern Insights
        report_lines.append("\n🧠 PATTERN INSIGHTS")
        report_lines.append("-"*40)
        
        insights = self._generate_pattern_insights()
        for insight in insights:
            report_lines.append(f"• {insight}")
        
        # Action Plan
        report_lines.append("\n🎯 RECOMMENDED ACTION PLAN")
        report_lines.append("-"*40)
        
        actions = self._generate_action_plan()
        for action in actions:
            report_lines.append(f"• {action}")
        
        report_lines.append("")
        report_lines.append("="*80)
        report_lines.append("End of Bottleneck Report")
        report_lines.append("="*80)
        
        report = "\n".join(report_lines)
        
        # Save report
        self._save_report_to_file(report)
        
        return report
    
    def _generate_pattern_insights(self) -> List[str]:
        """Generate pattern-based insights from bottlenecks."""
        insights = []
        
        if not self.bottlenecks:
            insights.append("No significant patterns detected today.")
            return insights
        
        # Group by bottleneck type
        by_type = defaultdict(list)
        for bottleneck in self.bottlenecks:
            by_type[bottleneck.bottleneck_type].append(bottleneck)
        
        # Generate insights for each type
        for b_type, bottlenecks in by_type.items():
            if bottlenecks:
                # Time distribution
                times = [b.start_time.hour for b in bottlenecks]
                if times:
                    avg_hour = np.mean(times)
                    peak_hour = max(set(times), key=times.count)
                    
                    if b_type == 'understaffed':
                        insights.append(
                            f"Understaffing tends to occur around {int(peak_hour):02d}:00 "
                            f"(average {avg_hour:.1f} hours)"
                        )
                    elif b_type == 'blockage':
                        insights.append(
                            f"Blockages peak at {int(peak_hour):02d}:00, "
                            f"suggesting layout issues during busy periods"
                        )
        
        # Overall insights
        total_duration = sum(b.duration_minutes for b in self.bottlenecks)
        insights.append(f"Total bottleneck time: {total_duration:.1f} minutes ({total_duration/60:.1f} hours)")
        
        if total_duration > 120:  # More than 2 hours
            insights.append("⚠ Significant operational inefficiencies detected")
        
        return insights
    
    def _generate_action_plan(self) -> List[str]:
        """Generate actionable recommendations."""
        actions = []
        
        if not self.bottlenecks:
            actions.append("No specific actions needed based on today's data.")
            return actions
        
        # Count critical/high severity bottlenecks
        critical_bottlenecks = [b for b in self.bottlenecks 
                               if b.severity in ['critical', 'high']]
        
        if critical_bottlenecks:
            actions.append("IMMEDIATE ACTIONS REQUIRED:")
            for bottleneck in critical_bottlenecks[:3]:  # Top 3 most critical
                actions.append(f"  - Address {bottleneck.bottleneck_type} in {bottleneck.zone_name or 'affected area'}")
        
        # General recommendations
        actions.append("\nSTRATEGIC RECOMMENDATIONS:")
        actions.append("  1. Review staffing schedules for peak bottleneck periods")
        actions.append("  2. Analyze physical layout where blockages occur")
        actions.append("  3. Implement monitoring for recurring bottleneck patterns")
        actions.append("  4. Consider workflow optimization in high-traffic zones")
        
        # Specific recommendations based on bottleneck types
        bottleneck_types = set(b.bottleneck_type for b in self.bottlenecks)
        
        if 'understaffed' in bottleneck_types:
            actions.append("  5. Evaluate staffing levels during high-activity periods")
        
        if 'blockage' in bottleneck_types:
            actions.append("  6. Review customer flow and queuing systems")
        
        if 'choke_point' in bottleneck_types:
            actions.append("  7. Consider expanding or redesigning crowded zones")
        
        if 'idle_cluster' in bottleneck_types:
            actions.append("  8. Implement activity monitoring and task management")
        
        return actions
    
    def _save_report_to_file(self, report: str):
        """Save bottleneck report to file."""
        filename = f"bottleneck_report_{self.today_str}.txt"
        filepath = os.path.join(self.reports_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\n✅ Bottleneck report saved to: {filepath}")
        except Exception as e:
            print(f"❌ Error saving report: {e}")
    
    def visualize_bottlenecks(self):
        """Create visualization of bottlenecks (optional)."""
        try:
            if not self.timeline_data or not self.bottlenecks:
                print("⚠ No data for visualization")
                return
            
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot 1: Timeline of motion and face counts
            times = [d['timestamp'] for d in self.timeline_data]
            motions = [d['avg_motion'] for d in self.timeline_data]
            faces = [d['face_count'] for d in self.timeline_data]
            
            ax1.plot(times, motions, 'b-', label='Motion Score', alpha=0.7)
            ax1.set_ylabel('Motion Score', color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            ax1.grid(True, alpha=0.3)
            
            ax1_twin = ax1.twinx()
            ax1_twin.plot(times, faces, 'r-', label='Face Count', alpha=0.7)
            ax1_twin.set_ylabel('Face Count', color='r')
            ax1_twin.tick_params(axis='y', labelcolor='r')
            
            ax1.set_title('Motion vs Face Count Timeline')
            ax1.legend(loc='upper left')
            ax1_twin.legend(loc='upper right')
            
            # Plot 2: Bottleneck visualization
            ax2.set_title('Detected Bottlenecks')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Bottleneck Type')
            ax2.set_yticks([1, 2, 3, 4])
            ax2.set_yticklabels(['Understaffed', 'Blockage', 'Choke Point', 'Idle Cluster'])
            
            colors = {'critical': 'red', 'high': 'orange', 'medium': 'yellow', 'low': 'green'}
            
            for i, bottleneck in enumerate(self.bottlenecks):
                y_position = {'understaffed': 1, 'blockage': 2, 
                             'choke_point': 3, 'idle_cluster': 4}[bottleneck.bottleneck_type]
                
                ax2.plot([bottleneck.start_time, bottleneck.end_time],
                        [y_position, y_position],
                        linewidth=3,
                        color=colors.get(bottleneck.severity, 'gray'),
                        alpha=0.7,
                        label=f"{bottleneck.severity} severity" if i == 0 else "")
                
                # Add text label for duration
                mid_time = bottleneck.start_time + (bottleneck.end_time - bottleneck.start_time) / 2
                ax2.text(mid_time, y_position + 0.1,
                        f"{bottleneck.duration_minutes:.0f}m",
                        ha='center', va='bottom', fontsize=8)
            
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            plt.tight_layout()
            
            # Save visualization
            vis_filename = f"bottleneck_visualization_{self.today_str}.png"
            vis_filepath = os.path.join(self.reports_dir, vis_filename)
            plt.savefig(vis_filepath, dpi=150)
            plt.close()
            
            print(f"✅ Visualization saved to: {vis_filepath}")
            
        except Exception as e:
            print(f"⚠ Error creating visualization: {e}")
    
    def export_to_json(self):
        """Export bottlenecks to JSON format for external analysis."""
        if not self.bottlenecks:
            return
        
        bottlenecks_data = []
        for bottleneck in self.bottlenecks:
            bottlenecks_data.append({
                'start_time': bottleneck.start_time.isoformat(),
                'end_time': bottleneck.end_time.isoformat(),
                'bottleneck_type': bottleneck.bottleneck_type,
                'severity': bottleneck.severity,
                'duration_minutes': bottleneck.duration_minutes,
                'avg_motion_score': bottleneck.avg_motion_score,
                'avg_face_count': bottleneck.avg_face_count,
                'confidence': bottleneck.confidence,
                'zone_name': bottleneck.zone_name,
                'description': bottleneck.description,
                'recommendations': bottleneck.recommendations
            })
        
        json_filename = f"bottlenecks_{self.today_str}.json"
        json_filepath = os.path.join(self.reports_dir, json_filename)
        
        try:
            with open(json_filepath, 'w', encoding='utf-8') as f:
                json.dump(bottlenecks_data, f, indent=2, default=str)
            print(f"✅ Bottlenecks exported to JSON: {json_filepath}")
        except Exception as e:
            print(f"❌ Error exporting to JSON: {e}")


def main():
    """Main function to run the Bottleneck Detector."""
    print("\n" + "="*80)
    print("🧭 INVISIBLE BOTTLENECK DETECTOR")
    print("="*80)
    print("\nThis system analyzes your Picam data to detect invisible bottlenecks:")
    print("• Understaffing: High activity with insufficient staff")
    print("• Blockages: Crowding with low movement")
    print("• Choke Points: Sustained crowding in specific zones")
    print("• Idle Clusters: Multiple idle alerts in short periods")
    print("\n" + "="*80)
    
    # Initialize detector
    detector = BottleneckDetector()
    
    # Load data
    if not detector.load_and_prepare_data():
        print("\n❌ Failed to load data. Please ensure:")
        print("   1. Video processor has been run today")
        print("   2. CSV files exist in 'picam_reports' folder")
        print("   3. Today's date is correct")
        return
    
    # Detect bottlenecks
    bottlenecks = detector.detect_bottlenecks()
    
    if not bottlenecks:
        print("\n✅ No bottlenecks detected today!")
        print("   Your operations appear to be running smoothly.")
        return
    
    # Generate and display report
    report = detector.generate_bottleneck_report()
    print("\n" + report)
    
    # Ask about visualization
    choice = input("\nGenerate visualization? (y/n): ").lower()
    if choice == 'y':
        detector.visualize_bottlenecks()
    
    # Ask about JSON export
    choice = input("\nExport to JSON format? (y/n): ").lower()
    if choice == 'y':
        detector.export_to_json()
    
    print("\n" + "="*80)
    print("🎯 BOTTLENECK ANALYSIS COMPLETE")
    print("="*80)
    print("\nKey takeaways:")
    print(f"• Detected {len(bottlenecks)} total bottlenecks")
    
    # Count by severity
    severities = defaultdict(int)
    for b in bottlenecks:
        severities[b.severity] += 1
    
    if severities.get('critical', 0) > 0:
        print(f"• ⚠ {severities['critical']} CRITICAL bottlenecks need immediate attention")
    if severities.get('high', 0) > 0:
        print(f"• ⚠ {severities['high']} HIGH severity bottlenecks should be addressed")
    
    print("\nCheck the generated report for detailed analysis and recommendations.")


if __name__ == "__main__":
    main()