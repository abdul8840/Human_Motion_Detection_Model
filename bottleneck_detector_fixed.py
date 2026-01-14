# bottleneck_detector_fixed.py
import csv
import os
import numpy as np
from datetime import datetime, date, timedelta
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Any, Optional
import json

class BottleneckDetectorFixed:
    """Fixed Bottleneck Detector with PASSIVE bottleneck detection."""
    
    def __init__(self, reports_dir: str = "picam_reports"):
        self.reports_dir = reports_dir
        self.today = date.today()
        self.today_str = self.today.strftime("%Y%m%d")
        
        # Optimized configuration
        self.config = {
            'time_window_minutes': 15,
            'high_motion_threshold': 0.25,     # For understaffed detection
            'very_low_motion_threshold': 0.05,  # For traditional blockage
            'passive_motion_max': 0.08,         # NEW: For passive bottlenecks
            'high_face_threshold': 3,
            'low_face_threshold': 1,
            'min_duration_minutes': 10
        }
        
        self.bottlenecks = []
    
    def analyze_today(self):
        """Main analysis function."""
        print(f"\n{'='*60}")
        print("BOTTLENECK ANALYSIS WITH PASSIVE DETECTION")
        print(f"Date: {self.today.strftime('%Y-%m-%d')}")
        print(f"{'='*60}")
        
        # Load all data
        data = self._load_all_data()
        
        if not data['detections']:
            print("No detection data found.")
            return
        
        print(f"Detections loaded: {len(data['detections'])}")
        print(f"Motion events: {len(data['motion_events'])}")
        print(f"Alerts: {len(data['alerts'])}")
        
        # Create hourly analysis
        hourly_data = self._create_hourly_analysis(data)
        
        # Detect bottlenecks (now includes passive)
        bottlenecks = self._detect_all_bottlenecks(hourly_data, data['alerts'])
        
        if not bottlenecks:
            print("\n✅ No significant bottlenecks detected.")
            print("Your operations appear efficient today!")
            return
        
        # Generate and save report
        report = self._generate_report(bottlenecks, hourly_data)
        print(f"\n{'='*60}")
        print("BOTTLENECK REPORT (WITH PASSIVE DETECTION)")
        print(f"{'='*60}")
        print(report)
        
        # Save to file
        self._save_report(report, bottlenecks)
    
    def _load_all_data(self) -> Dict[str, List]:
        """Load all CSV data."""
        data = {
            'detections': [],
            'motion_events': [],
            'alerts': [],
            'zone_stats': []
        }
        
        # File paths
        files = {
            'detections': f"detections_{self.today_str}.csv",
            'motion_events': f"motion_events_{self.today_str}.csv",
            'alerts': f"alerts_{self.today_str}.csv",
            'zone_stats': f"zone_stats_{self.today_str}.csv"
        }
        
        for data_type, filename in files.items():
            filepath = os.path.join(self.reports_dir, filename)
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        data[data_type] = list(reader)
                    print(f"✓ Loaded {len(data[data_type])} {data_type}")
                except Exception as e:
                    print(f"⚠ Error loading {data_type}: {e}")
            else:
                print(f"⚠ File not found: {filename}")
        
        return data
    
    def _create_hourly_analysis(self, data: Dict) -> Dict:
        """Create hourly aggregated data."""
        hourly_stats = {}
        
        # Process detections by hour
        for detection in data['detections']:
            try:
                timestamp = datetime.strptime(
                    detection['timestamp'].split('.')[0], 
                    "%Y-%m-%d %H:%M:%S"
                )
                hour_key = timestamp.replace(minute=0, second=0, microsecond=0)
                
                if hour_key not in hourly_stats:
                    hourly_stats[hour_key] = {
                        'face_count': 0,
                        'person_count': 0,
                        'motion_scores': [],
                        'zones': defaultdict(int),
                        'detection_count': 0
                    }
                
                # Count faces and persons
                if detection['object_type'] == 'face':
                    hourly_stats[hour_key]['face_count'] += 1
                elif detection['object_type'] == 'person':
                    hourly_stats[hour_key]['person_count'] += 1
                
                # Track zones
                if detection['zone_name']:
                    hourly_stats[hour_key]['zones'][detection['zone_name']] += 1
                
                hourly_stats[hour_key]['detection_count'] += 1
                
            except Exception:
                continue
        
        # Add motion data
        for motion_event in data['motion_events']:
            try:
                timestamp = datetime.strptime(
                    motion_event['timestamp'].split('.')[0], 
                    "%Y-%m-%d %H:%M:%S"
                )
                hour_key = timestamp.replace(minute=0, second=0, microsecond=0)
                
                if hour_key in hourly_stats:
                    motion_score = float(motion_event.get('motion_score', 0))
                    hourly_stats[hour_key]['motion_scores'].append(motion_score)
            except Exception:
                continue
        
        # Calculate averages
        for hour_key, stats in hourly_stats.items():
            if stats['motion_scores']:
                stats['avg_motion'] = np.mean(stats['motion_scores'])
                stats['max_motion'] = max(stats['motion_scores'])
            else:
                stats['avg_motion'] = 0
                stats['max_motion'] = 0
            
            # Find busiest zone
            if stats['zones']:
                stats['busiest_zone'] = max(stats['zones'].items(), key=lambda x: x[1])[0]
                stats['busiest_zone_count'] = stats['zones'][stats['busiest_zone']]
            else:
                stats['busiest_zone'] = None
                stats['busiest_zone_count'] = 0
        
        return hourly_stats
    
    def _detect_all_bottlenecks(self, hourly_data: Dict, alerts: List) -> List[Dict]:
        """Detect all types of bottlenecks, including PASSIVE."""
        bottlenecks = []
        
        print(f"\nAnalyzing {len(hourly_data)} hours of data...")
        
        # 1. Detect understaffed hours (high motion, low faces)
        understaffed = self._detect_understaffed_hours(hourly_data)
        bottlenecks.extend(understaffed)
        
        # 2. Detect traditional blockage hours (very low motion, high faces)
        blockages = self._detect_blockage_hours(hourly_data)
        bottlenecks.extend(blockages)
        
        # 3. Detect PASSIVE bottlenecks (low motion, high faces) - NEW
        passive_bottlenecks = self._detect_passive_bottlenecks(hourly_data)
        bottlenecks.extend(passive_bottlenecks)
        
        # 4. Detect choke points (zones with sustained high occupancy)
        choke_points = self._detect_choke_points(hourly_data)
        bottlenecks.extend(choke_points)
        
        # 5. Detect idle clusters from alerts
        idle_clusters = self._detect_idle_clusters(alerts)
        bottlenecks.extend(idle_clusters)
        
        # Sort by severity (passive bottlenecks get priority as they're more insidious)
        for bottleneck in bottlenecks:
            if bottleneck['type'] == 'passive_bottleneck':
                bottleneck['severity_score'] *= 1.5  # Boost passive bottleneck severity
        
        bottlenecks.sort(key=lambda x: x.get('severity_score', 0), reverse=True)
        
        return bottlenecks
    
    def _detect_understaffed_hours(self, hourly_data: Dict) -> List[Dict]:
        """Detect hours with high motion but low face count."""
        bottlenecks = []
        
        for hour, stats in hourly_data.items():
            if (stats['avg_motion'] > self.config['high_motion_threshold'] and
                stats['face_count'] <= self.config['low_face_threshold']):
                
                severity_score = stats['avg_motion'] * 10
                
                bottlenecks.append({
                    'type': 'understaffed',
                    'hour': hour.strftime('%H:%M'),
                    'duration': '1 hour',
                    'motion_score': round(stats['avg_motion'], 3),
                    'face_count': stats['face_count'],
                    'severity': self._get_severity(severity_score),
                    'severity_score': severity_score,
                    'description': f"High activity ({stats['avg_motion']:.3f} motion) with only {stats['face_count']} faces",
                    'behavioral_truth': "Staff overwhelmed, tasks not completed",
                    'recommendations': [
                        "Consider adding staff during this hour",
                        "Review task allocation",
                        "Monitor for workflow bottlenecks"
                    ]
                })
        
        return bottlenecks
    
    def _detect_blockage_hours(self, hourly_data: Dict) -> List[Dict]:
        """Detect hours with high face count but VERY low motion (<0.05)."""
        bottlenecks = []
        
        for hour, stats in hourly_data.items():
            if (stats['face_count'] >= self.config['high_face_threshold'] and
                stats['avg_motion'] < self.config['very_low_motion_threshold']):
                
                severity_score = stats['face_count'] * 3
                
                bottlenecks.append({
                    'type': 'blockage',
                    'hour': hour.strftime('%H:%M'),
                    'duration': '1 hour',
                    'face_count': stats['face_count'],
                    'motion_score': round(stats['avg_motion'], 3),
                    'severity': self._get_severity(severity_score),
                    'severity_score': severity_score,
                    'description': f"Severe congestion: {stats['face_count']} faces, extremely low movement ({stats['avg_motion']:.3f} motion)",
                    'behavioral_truth': "Clear obstruction or complete stoppage",
                    'recommendations': [
                        "Review layout and flow - possible physical blockage",
                        "Consider queuing systems",
                        "Check for equipment failures"
                    ]
                })
        
        return bottlenecks
    
    def _detect_passive_bottlenecks(self, hourly_data: Dict) -> List[Dict]:
        """NEW: Detect passive bottlenecks (0.05-0.08 motion with high faces)."""
        bottlenecks = []
        
        PASSIVE_FACE_THRESHOLD = 3
        PASSIVE_MOTION_MIN = 0.05
        PASSIVE_MOTION_MAX = self.config['passive_motion_max']
        
        for hour, stats in hourly_data.items():
            # People present but little happening (silent value loss)
            if (stats['face_count'] >= PASSIVE_FACE_THRESHOLD and
                PASSIVE_MOTION_MIN <= stats['avg_motion'] < PASSIVE_MOTION_MAX):
                
                # Higher severity weight - passive bottlenecks are insidious
                severity_score = stats['face_count'] * 5
                
                bottlenecks.append({
                    'type': 'passive_bottleneck',
                    'hour': hour.strftime('%H:%M'),
                    'duration': '1 hour',
                    'face_count': stats['face_count'],
                    'motion_score': round(stats['avg_motion'], 3),
                    'severity': self._get_severity(severity_score),
                    'severity_score': severity_score,
                    'behavioral_truth': "PEOPLE PRESENT BUT NOT MOVING - VALUE SILENTLY LEAKING",
                    'description': f"⚠️ SILENT CONGESTION: {stats['face_count']} faces present, "
                                 f"minimal movement ({stats['avg_motion']:.3f} motion)",
                    'likely_causes': [
                        "Decision paralysis / confusion",
                        "Waiting for unclear instructions",
                        "Uncertain workflow next steps",
                        "Silent coordination breakdown"
                    ],
                    'recommendations': [
                        "IMMEDIATE: Investigate silent waiting/confusion",
                        "Review process clarity and instructions",
                        "Implement activity checkpoints every 15 minutes",
                        "Assign clear task ownership",
                        "This is often decision paralysis - provide guidance"
                    ],
                    'priority': 'HIGH'  # Explicit priority flag
                })
        
        return bottlenecks
    
    def _detect_choke_points(self, hourly_data: Dict) -> List[Dict]:
        """Detect zones with sustained high occupancy."""
        bottlenecks = []
        zone_occupancy = defaultdict(list)
        
        # Collect zone data across all hours
        for hour, stats in hourly_data.items():
            for zone, count in stats['zones'].items():
                zone_occupancy[zone].append({
                    'hour': hour,
                    'count': count,
                    'motion': stats['avg_motion']
                })
        
        # Analyze each zone
        for zone, data_points in zone_occupancy.items():
            if len(data_points) < 3:  # Need data from at least 3 hours
                continue
            
            # Check for sustained high occupancy
            high_occupancy_hours = [dp for dp in data_points if dp['count'] >= 3]
            
            if len(high_occupancy_hours) >= 2:  # At least 2 hours of high occupancy
                avg_count = np.mean([dp['count'] for dp in high_occupancy_hours])
                hours_str = ", ".join(sorted(set(dp['hour'].strftime('%H:%M') for dp in high_occupancy_hours)))
                
                bottlenecks.append({
                    'type': 'choke_point',
                    'zone': zone,
                    'hours': hours_str,
                    'avg_occupancy': round(avg_count, 1),
                    'duration': f"{len(high_occupancy_hours)} hours",
                    'severity': 'high' if avg_count >= 5 else 'medium',
                    'severity_score': avg_count * 2,
                    'description': f"Sustained crowding in {zone} zone (avg {avg_count:.1f} people)",
                    'recommendations': [
                        f"Review {zone} layout and capacity",
                        "Consider expanding or redesigning this area",
                        "Implement scheduling for high-demand periods"
                    ]
                })
        
        return bottlenecks
    
    def _detect_idle_clusters(self, alerts: List) -> List[Dict]:
        """Detect clusters of idle alerts."""
        bottlenecks = []
        idle_alerts = []
        
        # Filter idle alerts
        for alert in alerts:
            if alert.get('alert_type') == 'idle_staff':
                try:
                    timestamp = datetime.strptime(
                        alert['timestamp'].split('.')[0], 
                        "%Y-%m-%d %H:%M:%S"
                    )
                    idle_alerts.append({
                        'time': timestamp,
                        'zone': alert.get('zone_name', 'Unknown'),
                        'duration': float(alert.get('duration_seconds', 0))
                    })
                except Exception:
                    continue
        
        if len(idle_alerts) >= 3:
            # Group by time proximity (within 2 hours)
            idle_alerts.sort(key=lambda x: x['time'])
            
            clusters = []
            current_cluster = []
            
            for alert in idle_alerts:
                if not current_cluster:
                    current_cluster.append(alert)
                else:
                    last_alert = current_cluster[-1]
                    time_diff = (alert['time'] - last_alert['time']).total_seconds() / 3600
                    
                    if time_diff <= 2:  # Within 2 hours
                        current_cluster.append(alert)
                    else:
                        if len(current_cluster) >= 3:
                            clusters.append(current_cluster)
                        current_cluster = [alert]
            
            # Check last cluster
            if len(current_cluster) >= 3:
                clusters.append(current_cluster)
            
            # Create bottlenecks for each cluster
            for i, cluster in enumerate(clusters, 1):
                start_time = cluster[0]['time'].strftime('%H:%M')
                end_time = cluster[-1]['time'].strftime('%H:%M')
                zones = [a['zone'] for a in cluster]
                zone_counts = {zone: zones.count(zone) for zone in set(zones)}
                most_common_zone = max(zone_counts.items(), key=lambda x: x[1])[0] if zone_counts else "Various"
                
                bottlenecks.append({
                    'type': 'idle_cluster',
                    'time_range': f"{start_time} - {end_time}",
                    'alert_count': len(cluster),
                    'zones': most_common_zone,
                    'duration': f"{(cluster[-1]['time'] - cluster[0]['time']).total_seconds()/3600:.1f} hours",
                    'severity': 'high' if len(cluster) >= 5 else 'medium',
                    'severity_score': len(cluster) * 2,
                    'description': f"Cluster of {len(cluster)} idle alerts in {most_common_zone}",
                    'recommendations': [
                        "Review task management and supervision",
                        "Implement activity monitoring",
                        "Consider workload redistribution"
                    ]
                })
        
        return bottlenecks
    
    def _get_severity(self, score: float) -> str:
        """Convert score to severity level."""
        if score >= 25:
            return 'CRITICAL'
        elif score >= 18:
            return 'HIGH'
        elif score >= 12:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _generate_report(self, bottlenecks: List[Dict], hourly_data: Dict) -> str:
        """Generate comprehensive bottleneck report."""
        report_lines = []
        
        report_lines.append(f"BOTTLENECK ANALYSIS REPORT (WITH PASSIVE DETECTION)")
        report_lines.append(f"Date: {self.today.strftime('%Y-%m-%d')}")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")  # Full timestamp
        report_lines.append(f"="*60)
        
        # Summary
        report_lines.append(f"\n📊 SUMMARY")
        report_lines.append(f"-"*40)
        report_lines.append(f"Total hours analyzed: {len(hourly_data)}")
        report_lines.append(f"Total bottlenecks detected: {len(bottlenecks)}")
        
        # Count by type (highlight passive bottlenecks)
        type_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        
        for bottleneck in bottlenecks:
            type_counts[bottleneck['type']] += 1
            severity_counts[bottleneck['severity']] += 1
        
        if type_counts:
            report_lines.append(f"\nBottleneck types:")
            for b_type, count in sorted(type_counts.items()):
                indicator = "⚠️ " if b_type == 'passive_bottleneck' else "• "
                report_lines.append(f"  {indicator}{b_type}: {count}")
        
        if severity_counts:
            report_lines.append(f"\nSeverity levels:")
            for severity, count in severity_counts.items():
                report_lines.append(f"  • {severity}: {count}")
        
        # Highlight passive bottlenecks first
        passive_bottlenecks = [b for b in bottlenecks if b['type'] == 'passive_bottleneck']
        if passive_bottlenecks:
            report_lines.append(f"\n🚨 CRITICAL FINDING: {len(passive_bottlenecks)} PASSIVE BOTTLENECKS DETECTED")
            report_lines.append(f"These represent SILENT VALUE LEAKAGE - people present but not productive")
            report_lines.append(f"-"*40)
        
        # Detailed bottlenecks (passive first, then others)
        if bottlenecks:
            report_lines.append(f"\n🔍 DETECTED BOTTLENECKS (PRIORITY ORDER)")
            report_lines.append(f"-"*40)
            
            # Sort: passive first, then by severity
            sorted_bottlenecks = sorted(
                bottlenecks,
                key=lambda x: (0 if x['type'] == 'passive_bottleneck' else 1, -x['severity_score'])
            )
            
            for i, bottleneck in enumerate(sorted_bottlenecks, 1):
                if bottleneck['type'] == 'passive_bottleneck':
                    report_lines.append(f"\n{i}. 🚨 {bottleneck['type'].upper()} ({bottleneck['severity']})")
                    report_lines.append(f"   ⏰ Time: {bottleneck.get('hour', bottleneck.get('time_range', 'N/A'))}")
                else:
                    report_lines.append(f"\n{i}. {bottleneck['type'].upper()} ({bottleneck['severity']})")
                    report_lines.append(f"   Time: {bottleneck.get('hour', bottleneck.get('time_range', bottleneck.get('hours', 'N/A')))}")
                
                report_lines.append(f"   Duration: {bottleneck['duration']}")
                
                if bottleneck['type'] in ['understaffed', 'blockage', 'passive_bottleneck']:
                    report_lines.append(f"   Motion Score: {bottleneck.get('motion_score', 'N/A')}")
                    report_lines.append(f"   Face Count: {bottleneck.get('face_count', 'N/A')}")
                
                if 'zone' in bottleneck:
                    report_lines.append(f"   Zone: {bottleneck['zone']}")
                
                report_lines.append(f"   Description: {bottleneck['description']}")
                
                if 'behavioral_truth' in bottleneck:
                    report_lines.append(f"   Behavioral Truth: {bottleneck['behavioral_truth']}")
                
                report_lines.append(f"   Recommendations:")
                for rec in bottleneck['recommendations'][:3]:
                    report_lines.append(f"     • {rec}")
        
        # Overall recommendations
        report_lines.append(f"\n🎯 OVERALL RECOMMENDATIONS")
        report_lines.append(f"-"*40)
        
        if not bottlenecks:
            report_lines.append("No specific actions needed. Operations appear efficient.")
        else:
            # Generate overall recommendations, prioritizing passive bottleneck fixes
            all_recommendations = []
            for bottleneck in bottlenecks:
                if bottleneck['type'] == 'passive_bottleneck':
                    # Add all passive bottleneck recs first
                    all_recommendations.extend(bottleneck['recommendations'])
                else:
                    all_recommendations.extend(bottleneck['recommendations'][:2])
            
            # Get unique recommendations
            unique_recs = []
            for rec in all_recommendations:
                if rec not in unique_recs:
                    unique_recs.append(rec)
            
            for i, rec in enumerate(unique_recs[:5], 1):
                priority = "🚨 " if any(word in rec.lower() for word in ['immediate', 'critical', 'silent', 'passive']) else ""
                report_lines.append(f"{i}. {priority}{rec}")
        
        report_lines.append(f"\n" + "="*60)
        report_lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("End of Report")
        
        return "\n".join(report_lines)
    
    def _save_report(self, report: str, bottlenecks: List[Dict]):
        """Save report to file."""
        filename = f"bottleneck_report_detailed_{self.today_str}.txt"
        filepath = os.path.join(self.reports_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"✅ Report saved to: {filepath}")
            
            # Also save bottlenecks as JSON for later analysis
            json_file = os.path.join(self.reports_dir, f"bottlenecks_{self.today_str}.json")
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(bottlenecks, f, indent=2, default=str)
            print(f"✅ Bottlenecks saved to JSON: {json_file}")
            
        except Exception as e:
            print(f"⚠ Error saving report: {e}")


def main():
    """Main function."""
    print(f"\n{'='*60}")
    print("ADVANCED BOTTLENECK DETECTOR WITH PASSIVE DETECTION")
    print(f"{'='*60}")
    print("Now detects silent value leakage (passive bottlenecks)")
    
    detector = BottleneckDetectorFixed()
    detector.analyze_today()


if __name__ == "__main__":
    main()