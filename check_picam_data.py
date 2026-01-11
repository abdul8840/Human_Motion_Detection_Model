# check_picam_data.py
import csv
import os
from datetime import datetime, date
from collections import defaultdict

def check_picam_data():
    """Diagnostic tool to check Picam data structure."""
    today = date.today().strftime("%Y%m%d")
    reports_dir = "picam_reports"
    
    print(f"\n{'='*60}")
    print("PICAM DATA DIAGNOSTIC TOOL")
    print(f"Date: {today}")
    print(f"Reports Directory: {reports_dir}")
    print(f"{'='*60}")
    
    # Check if directory exists
    if not os.path.exists(reports_dir):
        print(f"❌ Directory '{reports_dir}' not found!")
        print(f"Please run 'python picam_with_csv.py' first to generate data.")
        return
    
    # Check each CSV file
    files_to_check = [
        f"detections_{today}.csv",
        f"motion_events_{today}.csv",
        f"alerts_{today}.csv",
        f"daily_stats_{today}.csv",
        f"zone_stats_{today}.csv"
    ]
    
    for filename in files_to_check:
        filepath = os.path.join(reports_dir, filename)
        
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    
                    print(f"\n📄 {filename}:")
                    print(f"   ✓ File exists")
                    print(f"   ✓ Rows: {len(rows)}")
                    
                    if rows:
                        print(f"   ✓ Columns: {', '.join(rows[0].keys())}")
                        
                        # Sample first row
                        print(f"   ✓ Sample data:")
                        for key, value in list(rows[0].items())[:3]:  # First 3 columns
                            print(f"     {key}: {value[:50]}{'...' if len(str(value)) > 50 else ''}")
                        
                        # Check timestamps
                        if 'timestamp' in rows[0]:
                            timestamps = [row['timestamp'] for row in rows[:5]]
                            print(f"   ✓ First 5 timestamps:")
                            for ts in timestamps:
                                print(f"     {ts}")
                    
            except Exception as e:
                print(f"   ❌ Error reading file: {e}")
        else:
            print(f"\n📄 {filename}:")
            print(f"   ❌ File not found")
    
    # Analyze detections data
    detections_file = os.path.join(reports_dir, f"detections_{today}.csv")
    if os.path.exists(detections_file):
        print(f"\n{'='*60}")
        print("DETECTIONS ANALYSIS")
        print(f"{'='*60}")
        
        with open(detections_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            detections = list(reader)
            
            # Group by hour
            hourly_counts = defaultdict(int)
            object_types = defaultdict(int)
            zones = defaultdict(int)
            
            for detection in detections:
                try:
                    # Parse timestamp
                    timestamp = datetime.strptime(
                        detection['timestamp'].split('.')[0], 
                        "%Y-%m-%d %H:%M:%S"
                    )
                    hour_key = timestamp.replace(minute=0, second=0, microsecond=0)
                    hourly_counts[hour_key] += 1
                    
                    # Count object types
                    obj_type = detection.get('object_type', 'unknown')
                    object_types[obj_type] += 1
                    
                    # Count zones
                    zone = detection.get('zone_name', 'unknown')
                    if zone:
                        zones[zone] += 1
                        
                except Exception:
                    continue
            
            print(f"Total detections: {len(detections)}")
            print(f"\nDetections by hour:")
            for hour, count in sorted(hourly_counts.items()):
                print(f"  {hour.strftime('%H:%M')}: {count} detections")
            
            print(f"\nObject types:")
            for obj_type, count in object_types.items():
                print(f"  {obj_type}: {count}")
            
            print(f"\nZones detected:")
            for zone, count in zones.items():
                print(f"  {zone}: {count}")
    
    print(f"\n{'='*60}")
    print("DIAGNOSTIC COMPLETE")
    print(f"{'='*60}")
    
    # Recommendations
    print(f"\n🎯 RECOMMENDATIONS:")
    
    # Check if data exists
    detections_file = os.path.join(reports_dir, f"detections_{today}.csv")
    if os.path.exists(detections_file):
        with open(detections_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            detections = list(reader)
            
            if len(detections) < 100:
                print(f"1. ⚠ Low data volume: Only {len(detections)} detections")
                print(f"   Run the video processor longer to collect more data")
            else:
                print(f"1. ✓ Sufficient data: {len(detections)} detections")
                print(f"   Ready for bottleneck analysis")
    
    # Check motion events
    motion_file = os.path.join(reports_dir, f"motion_events_{today}.csv")
    if os.path.exists(motion_file):
        with open(motion_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            motion_events = list(reader)
            
            if len(motion_events) > 0:
                print(f"2. ✓ Motion data available: {len(motion_events)} events")
            else:
                print(f"2. ⚠ No motion events found")
                print(f"   Check if motion detection is enabled in settings")


if __name__ == "__main__":
    check_picam_data()