# truth_report_generator.py (updated version)
import csv
import os
from datetime import datetime, date, timedelta
from collections import defaultdict
import pandas as pd
from typing import Dict, List, Tuple, Any

class TruthReportGenerator:
    """Generates daily 'Truth Report' from CSV data."""
    
    def __init__(self, reports_dir: str = "picam_reports"):
        """
        Initialize the report generator.
        
        Args:
            reports_dir: Directory containing CSV reports
        """
        self.reports_dir = reports_dir
        self.today = date.today()
        self.today_str = self.today.strftime("%Y%m%d")
        
    def load_today_data(self) -> Dict[str, Any]:
        """Load all today's data from CSV files."""
        data = {
            'detections': [],
            'alerts': [],
            'daily_stats': {},
            'zone_stats': [],
            'motion_events': []
        }
        
        try:
            # Load detections
            detections_file = os.path.join(self.reports_dir, f"detections_{self.today_str}.csv")
            if os.path.exists(detections_file):
                with open(detections_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    data['detections'] = list(reader)
            
            # Load alerts
            alerts_file = os.path.join(self.reports_dir, f"alerts_{self.today_str}.csv")
            if os.path.exists(alerts_file):
                with open(alerts_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    data['alerts'] = list(reader)
            
            # Load daily stats
            daily_stats_file = os.path.join(self.reports_dir, f"daily_stats_{self.today_str}.csv")
            if os.path.exists(daily_stats_file):
                with open(daily_stats_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row['date'] == self.today.strftime("%Y-%m-%d"):
                            data['daily_stats'] = row
                            break
            
            # Load zone stats
            zone_stats_file = os.path.join(self.reports_dir, f"zone_stats_{self.today_str}.csv")
            if os.path.exists(zone_stats_file):
                with open(zone_stats_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    data['zone_stats'] = [row for row in reader if row['date'] == self.today.strftime("%Y-%m-%d")]
            
            # Load motion events
            motion_file = os.path.join(self.reports_dir, f"motion_events_{self.today_str}.csv")
            if os.path.exists(motion_file):
                with open(motion_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    data['motion_events'] = list(reader)
                    
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            
        return data
    
    def calculate_total_unique_faces(self, detections: List[Dict]) -> int:
        """Calculate total unique faces seen today."""
        face_ids = set()
        
        for detection in detections:
            if detection['object_type'] == 'face' and detection['object_id']:
                try:
                    face_ids.add(int(detection['object_id']))
                except (ValueError, TypeError):
                    continue
        
        return len(face_ids)
    
    def calculate_counter_unattended_minutes(self, alerts: List[Dict], zone_stats: List[Dict]) -> float:
        """Calculate total minutes the Counter zone was unattended."""
        total_minutes = 0.0
        
        # Get Counter zone stats if available
        counter_stats = None
        for zone in zone_stats:
            if zone.get('zone_name') == 'Counter':
                counter_stats = zone
                break
        
        if counter_stats and 'total_occupancy_time' in counter_stats:
            try:
                # Calculate unattended time from occupancy time
                total_seconds_in_day = 24 * 60 * 60
                occupancy_seconds = float(counter_stats.get('total_occupancy_time', 0))
                unattended_seconds = total_seconds_in_day - occupancy_seconds
                total_minutes = max(0, unattended_seconds / 60)
            except (ValueError, TypeError):
                pass
        
        # Also check from alerts
        counter_alerts = [alert for alert in alerts 
                         if alert.get('alert_type') == 'unattended_station' 
                         and alert.get('zone_name') == 'Counter']
        
        for alert in counter_alerts:
            try:
                duration = float(alert.get('duration_seconds', 0))
                total_minutes += duration / 60
            except (ValueError, TypeError):
                continue
        
        return round(total_minutes, 2)
    
    def calculate_top_idle_times(self, alerts: List[Dict]) -> List[Tuple[str, float]]:
        """Calculate top 3 longest idle times."""
        idle_alerts = []
        
        for alert in alerts:
            if alert.get('alert_type') == 'idle_staff':
                try:
                    duration = float(alert.get('duration_seconds', 0))
                    zone_name = alert.get('zone_name', 'Unknown')
                    person_id = alert.get('object_id', 'Unknown')
                    
                    # Convert to minutes
                    minutes = duration / 60
                    
                    # Create description
                    description = f"Person {person_id} in {zone_name}"
                    idle_alerts.append((description, minutes))
                    
                except (ValueError, TypeError):
                    continue
        
        # Sort by duration (longest first) and get top 3
        idle_alerts.sort(key=lambda x: x[1], reverse=True)
        return idle_alerts[:3]
    
    def calculate_productivity_score(self, motion_events: List[Dict], detections: List[Dict]) -> float:
        """
        Calculate Productivity Score (0-100) based on:
        - Active Motion vs. Total Presence ratio
        """
        if not motion_events:
            return 50.0  # Default score if no data
        
        try:
            # Calculate average motion score
            total_motion = 0.0
            valid_motion_events = 0
            
            for event in motion_events:
                try:
                    motion_score = float(event.get('motion_score', 0))
                    total_motion += motion_score
                    valid_motion_events += 1
                except (ValueError, TypeError):
                    continue
            
            avg_motion = total_motion / valid_motion_events if valid_motion_events > 0 else 0
            
            # Calculate presence time from detections
            presence_minutes = 0
            detection_times = {}
            
            for detection in detections:
                try:
                    timestamp_str = detection.get('timestamp', '')
                    if timestamp_str:
                        # Parse timestamp
                        timestamp = datetime.strptime(timestamp_str.split('.')[0], "%Y-%m-%d %H:%M:%S")
                        hour = timestamp.hour
                        
                        # Track unique hours with detections
                        if hour not in detection_times:
                            detection_times[hour] = True
                except (ValueError, TypeError):
                    continue
            
            presence_minutes = len(detection_times) * 60  # Approximate minutes
            
            # Calculate productivity score
            # Based on motion activity and presence
            if presence_minutes > 0:
                # Motion component (0-70 points)
                motion_component = min(70, avg_motion * 70 * 10)  # Scale motion score
                
                # Presence component (0-30 points)
                # More presence is good, but not if it's all idle
                presence_hours = presence_minutes / 60
                presence_component = min(30, presence_hours * 2)  # Up to 15 hours gives max points
                
                productivity_score = motion_component + presence_component
            else:
                productivity_score = avg_motion * 50  # Base score on motion only
            
            # Ensure score is between 0-100
            return round(max(0, min(100, productivity_score)), 1)
            
        except Exception as e:
            print(f"⚠ Error calculating productivity score: {e}")
            return 50.0
    
    def calculate_zone_analytics(self, zone_stats: List[Dict]) -> Dict[str, Any]:
        """Calculate detailed zone analytics."""
        analytics = {
            'zones': [],
            'total_zone_entries': 0,
            'busiest_zone': None,
            'most_idle_zone': None
        }
        
        max_entries = 0
        max_idle = 0
        
        for zone in zone_stats:
            try:
                zone_name = zone.get('zone_name', 'Unknown')
                entries = int(zone.get('total_entries', 0))
                idle_alerts = int(zone.get('idle_alerts_count', 0))
                occupancy = float(zone.get('total_occupancy_time', 0))
                
                analytics['total_zone_entries'] += entries
                
                zone_data = {
                    'name': zone_name,
                    'entries': entries,
                    'idle_alerts': idle_alerts,
                    'occupancy_hours': round(occupancy / 3600, 2)
                }
                analytics['zones'].append(zone_data)
                
                # Track busiest zone
                if entries > max_entries:
                    max_entries = entries
                    analytics['busiest_zone'] = zone_name
                
                # Track most idle zone
                if idle_alerts > max_idle:
                    max_idle = idle_alerts
                    analytics['most_idle_zone'] = zone_name
                    
            except (ValueError, TypeError):
                continue
        
        return analytics
    
    def generate_report(self, save_to_file: bool = True) -> str:
        """Generate the complete Daily Truth Report."""
        
        print(f"\n{'='*60}")
        print(f"GENERATING DAILY TRUTH REPORT")
        print(f"Date: {self.today.strftime('%Y-%m-%d')}")
        print(f"{'='*60}")
        
        # Load data
        data = self.load_today_data()
        
        if not data['detections'] and not data['alerts']:
            print("⚠ No data found for today.")
            return "No data available for today."
        
        # Calculate metrics
        total_faces = self.calculate_total_unique_faces(data['detections'])
        counter_unattended = self.calculate_counter_unattended_minutes(data['alerts'], data['zone_stats'])
        top_idle_times = self.calculate_top_idle_times(data['alerts'])
        productivity_score = self.calculate_productivity_score(data['motion_events'], data['detections'])
        zone_analytics = self.calculate_zone_analytics(data['zone_stats'])
        
        # Get additional stats
        total_detections = len(data['detections'])
        total_alerts = len(data['alerts'])
        active_alerts = sum(1 for alert in data['alerts'] if alert.get('resolved') == 'False')
        
        # Build report
        report_lines = []
        report_lines.append("="*60)
        report_lines.append("DAILY TRUTH REPORT")
        report_lines.append(f"Date: {self.today.strftime('%B %d, %Y')}")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("="*60)
        report_lines.append("")
        
        # Summary Section
        report_lines.append("📊 EXECUTIVE SUMMARY")
        report_lines.append("-"*40)
        report_lines.append(f"• Productivity Score: {productivity_score}/100")
        report_lines.append(f"• Total Detections: {total_detections}")
        report_lines.append(f"• Unique Faces: {total_faces}")
        report_lines.append(f"• Total Alerts: {total_alerts} ({active_alerts} active)")
        report_lines.append("")
        
        # Zone Analytics
        report_lines.append("📍 ZONE ANALYTICS")
        report_lines.append("-"*40)
        report_lines.append(f"• Busiest Zone: {zone_analytics.get('busiest_zone', 'N/A')}")
        report_lines.append(f"• Total Zone Entries: {zone_analytics.get('total_zone_entries', 0)}")
        report_lines.append(f"• Most Idle Zone: {zone_analytics.get('most_idle_zone', 'N/A')}")
        report_lines.append("")
        
        for zone in zone_analytics['zones']:
            report_lines.append(f"  {zone['name']}:")
            report_lines.append(f"    • Entries: {zone['entries']}")
            report_lines.append(f"    • Occupancy: {zone['occupancy_hours']} hours")
            report_lines.append(f"    • Idle Alerts: {zone['idle_alerts']}")
        report_lines.append("")
        
        # Key Metrics (Features 12 & 21)
        report_lines.append("🔑 KEY METRICS")
        report_lines.append("-"*40)
        report_lines.append(f"1. Total Unique Faces: {total_faces}")
        report_lines.append(f"2. Counter Unattended Time: {counter_unattended} minutes")
        report_lines.append("")
        
        # Top Idle Times
        report_lines.append("⏰ TOP 3 LONGEST IDLE TIMES")
        report_lines.append("-"*40)
        if top_idle_times:
            for i, (description, minutes) in enumerate(top_idle_times, 1):
                report_lines.append(f"{i}. {description}: {minutes:.1f} minutes")
        else:
            report_lines.append("No idle alerts recorded today.")
        report_lines.append("")
        
        # Productivity Analysis
        report_lines.append("📈 PRODUCTIVITY ANALYSIS")
        report_lines.append("-"*40)
        report_lines.append(f"Score: {productivity_score}/100")
        
        # Add interpretation
        if productivity_score >= 80:
            report_lines.append("Interpretation: 👍 EXCELLENT productivity")
            report_lines.append("High activity levels with good presence.")
        elif productivity_score >= 60:
            report_lines.append("Interpretation: ✅ GOOD productivity")
            report_lines.append("Solid performance with room for improvement.")
        elif productivity_score >= 40:
            report_lines.append("Interpretation: ⚠️ AVERAGE productivity")
            report_lines.append("Moderate activity levels detected.")
        else:
            report_lines.append("Interpretation: ❌ LOW productivity")
            report_lines.append("Low activity and/or presence detected.")
        report_lines.append("")
        
        # Alert Summary
        if data['alerts']:
            report_lines.append("🚨 ALERT SUMMARY")
            report_lines.append("-"*40)
            
            alert_counts = defaultdict(int)
            for alert in data['alerts']:
                alert_type = alert.get('alert_type', 'Unknown')
                alert_counts[alert_type] += 1
            
            for alert_type, count in alert_counts.items():
                report_lines.append(f"• {alert_type}: {count}")
            report_lines.append("")
        
        # Recommendations
        report_lines.append("💡 RECOMMENDATIONS")
        report_lines.append("-"*40)
        
        if counter_unattended > 120:  # More than 2 hours unattended
            report_lines.append("• Consider staffing adjustments for Counter zone")
        
        if top_idle_times and top_idle_times[0][1] > 30:  # Idle more than 30 minutes
            report_lines.append("• Implement activity checks for long idle periods")
        
        if productivity_score < 60:
            report_lines.append("• Review workflow and engagement strategies")
        
        if zone_analytics.get('total_zone_entries', 0) < 10:
            report_lines.append("• Low traffic detected - consider promotion or outreach")
        
        report_lines.append("")
        report_lines.append("="*60)
        report_lines.append("End of Report")
        report_lines.append("="*60)
        
        # Combine into final report
        report = "\n".join(report_lines)
        
        # Print to console
        print(report)
        
        # Save to file
        if save_to_file:
            self.save_report_to_file(report)
        
        return report
    
    def save_report_to_file(self, report: str):
        """Save the report to a text file."""
        filename = f"truth_report_{self.today_str}.txt"
        filepath = os.path.join(self.reports_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\n✅ Truth Report saved to: {filepath}")
        except Exception as e:
            print(f"❌ Error saving report: {e}")
    
    def generate_html_report(self, report_text: str):
        """Generate an HTML version of the report (bonus feature)."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Daily Truth Report - {self.today.strftime('%Y-%m-%d')}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                .header {{ background-color: #2c3e50; color: white; padding: 20px; border-radius: 5px; margin-bottom: 30px; }}
                .section {{ margin: 30px 0; padding: 20px; border-left: 4px solid #3498db; background-color: #f8f9fa; }}
                .metric {{ background-color: #e8f4fc; padding: 15px; margin: 15px 0; border-radius: 5px; border: 1px solid #b3d7ff; }}
                .score {{ font-size: 28px; font-weight: bold; color: #27ae60; text-align: center; padding: 20px; }}
                .alert {{ color: #e74c3c; font-weight: bold; }}
                .good {{ color: #27ae60; }}
                .warning {{ color: #f39c12; }}
                .zone-item {{ margin: 10px 0; padding: 10px; background-color: #e8f4fc; border-radius: 3px; }}
                h1 {{ margin: 0; }}
                h3 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                ul {{ padding-left: 20px; }}
                li {{ margin: 8px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 Daily Truth Report</h1>
                <h3>Date: {self.today.strftime('%B %d, %Y')}</h3>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        """
        
        # Convert text report to HTML
        lines = report_text.split('\n')
        current_section = ""
        section_content = ""
        
        for line in lines:
            if line.strip() == "":
                continue
                
            if line.startswith("📊") or line.startswith("📍") or line.startswith("🔑") or \
               line.startswith("⏰") or line.startswith("📈") or line.startswith("🚨") or \
               line.startswith("💡") or line.startswith("DAILY TRUTH REPORT"):
                
                # Close previous section
                if current_section:
                    if "Productivity Score:" in section_content:
                        # Add score with special styling
                        html += f"""
                        <div class="section">
                            <h3>{current_section}</h3>
                            <div class="score">{section_content.replace('Productivity Score: ', '')}</div>
                        </div>
                        """
                    else:
                        html += f"""
                        <div class="section">
                            <h3>{current_section}</h3>
                            {section_content}
                        </div>
                        """
                
                # Start new section
                current_section = line
                section_content = ""
                
            elif line.startswith("•") or line.startswith("  "):
                # Add as list item or paragraph
                if line.strip().startswith("•"):
                    section_content += f"<li>{line.replace('•', '').strip()}</li>"
                else:
                    section_content += f"<p>{line.strip()}</p>"
                    
            elif "Score:" in line and "Interpretation:" not in line:
                section_content += f"<div class='metric'><strong>{line}</strong></div>"
                
            elif "Interpretation:" in line:
                section_content += f"<div class='metric'><strong>{line}</strong></div>"
                
            elif line.startswith("1.") or line.startswith("2.") or line.startswith("3."):
                section_content += f"<li>{line}</li>"
                
            elif not line.startswith("=") and not line.startswith("-") and line.strip():
                section_content += f"<p>{line}</p>"
        
        # Close the last section
        if current_section:
            if "Productivity Score:" in section_content:
                html += f"""
                <div class="section">
                    <h3>{current_section}</h3>
                    <div class="score">{section_content.replace('Productivity Score: ', '')}</div>
                </div>
                """
            else:
                html += f"""
                <div class="section">
                    <h3>{current_section}</h3>
                    {section_content}
                </div>
                """
        
        html += """
        </body>
        </html>
        """
        
        # Save HTML report
        html_filename = f"truth_report_{self.today_str}.html"
        html_filepath = os.path.join(self.reports_dir, html_filename)
        
        try:
            with open(html_filepath, 'w', encoding='utf-8') as f:
                f.write(html)
            print(f"✅ HTML Report saved to: {html_filepath}")
        except Exception as e:
            print(f"❌ Error saving HTML report: {e}")


def main():
    """Main function to generate the Truth Report."""
    print("\n" + "="*60)
    print("TRUTH REPORT GENERATOR")
    print("="*60)
    print("Analyzing today's data from CSV files...")
    
    # Create generator
    generator = TruthReportGenerator()
    
    # Generate report
    report = generator.generate_report(save_to_file=True)
    
    # Ask if user wants HTML version
    choice = input("\nGenerate HTML version? (y/n): ").lower()
    if choice == 'y':
        generator.generate_html_report(report)
    
    print("\n" + "="*60)
    print("Report generation complete!")
    print("="*60)


if __name__ == "__main__":
    main()