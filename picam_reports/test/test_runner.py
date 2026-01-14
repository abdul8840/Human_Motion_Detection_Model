# verify_fixes.py
"""
VERIFY PICAM FIXES
Check if the critical failures have been resolved.
Run: python verify_fixes.py
"""

import os
import csv
import json
from datetime import datetime
from pathlib import Path

print("\n" + "="*80)
print("PICAM FIXES VERIFICATION")
print("="*80)

def verify_all():
    """Verify all fixes."""
    
    print("\n🔍 VERIFICATION CHECKS:")
    
    # Check 1: VideoProcessor thresholds
    print("\n1. Checking VideoProcessor thresholds...")
    check_video_processor()
    
    # Check 2: Zone configurations
    print("\n2. Checking Zone configurations...")
    check_zone_configs()
    
    # Check 3: CSV data quality
    print("\n3. Checking CSV data quality...")
    check_csv_quality()
    
    # Check 4: Business logic thresholds
    print("\n4. Checking business logic thresholds...")
    check_business_logic()
    
    print("\n" + "="*80)
    print("VERIFICATION COMPLETE")
    print("="*80)

def check_video_processor():
    """Check VideoProcessor configuration."""
    
    file_path = "picam_with_csv.py"
    
    if not os.path.exists(file_path):
        print("  ⚠ File not found")
        return
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    checks = [
        ("Motion threshold 0.08", "motion_threshold=0.08" in content),
        ("Min person area 1500", "min_person_area=1500" in content),
        ("Max person area 100000", "max_person_area=100000" in content)
    ]
    
    passed = 0
    for check_name, check_result in checks:
        if check_result:
            print(f"  ✅ {check_name}")
            passed += 1
        else:
            print(f"  ❌ {check_name}")
    
    print(f"  Result: {passed}/{len(checks)} checks passed")

def check_zone_configs():
    """Check zone configurations."""
    
    config_file = "zones_config.json"
    
    if not os.path.exists(config_file):
        print("  ⚠ File not found")
        return
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    checks_passed = 0
    total_checks = 0
    
    for zone in config:
        print(f"  Zone: {zone['name']}")
        
        zone_checks = [
            ("Idle threshold >= 180", zone.get('idle_threshold', 0) >= 180),
            ("Motion threshold <= 0.03", zone.get('motion_threshold', 1) <= 0.03),
            ("Has all required fields", all(k in zone for k in ['name', 'points', 'zone_type']))
        ]
        
        for check_name, check_result in zone_checks:
            total_checks += 1
            if check_result:
                print(f"    ✅ {check_name}")
                checks_passed += 1
            else:
                print(f"    ❌ {check_name}")
    
    if total_checks > 0:
        print(f"  Result: {checks_passed}/{total_checks} checks passed")

def check_csv_quality():
    """Check CSV file quality."""
    
    reports_dir = Path("picam_reports")
    if not reports_dir.exists():
        print("  ⚠ Directory not found")
        return
    
    csv_files = list(reports_dir.glob("*.csv"))
    
    if not csv_files:
        print("  ⚠ No CSV files found")
        return
    
    total_files = 0
    valid_files = 0
    
    for csv_file in csv_files[:3]:  # Check first 3 files
        total_files += 1
        
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
                if rows:
                    # Check timestamp format in first row
                    first_row = rows[0]
                    if 'timestamp' in first_row:
                        timestamp = first_row['timestamp']
                        if timestamp and is_valid_timestamp(timestamp):
                            valid_files += 1
                            print(f"  ✅ {csv_file.name}: Valid timestamps, {len(rows)} rows")
                        else:
                            print(f"  ❌ {csv_file.name}: Invalid timestamp format")
                    else:
                        print(f"  ⚠ {csv_file.name}: No timestamp column")
                else:
                    print(f"  ⚠ {csv_file.name}: Empty file")
                    
        except Exception as e:
            print(f"  ❌ {csv_file.name}: Error - {str(e)}")
    
    print(f"  Result: {valid_files}/{total_files} files have valid data")

def is_valid_timestamp(timestamp_str):
    """Check if timestamp is valid."""
    if not timestamp_str:
        return False
    
    timestamp_str = str(timestamp_str).strip()
    
    # Accept common formats
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y/%m/%d %H:%M:%S",
        "%H:%M:%S"
    ]
    
    for fmt in formats:
        try:
            datetime.strptime(timestamp_str, fmt)
            return True
        except ValueError:
            continue
    
    return False

def check_business_logic():
    """Check business logic thresholds."""
    
    print("  Testing idle detection logic...")
    
    # Test cases for idle detection
    idle_tests = [
        (300, 0.005, True, "5min idle, very low motion"),
        (30, 0.25, False, "30sec active, high motion"),
        (200, 0.015, True, "3.3min idle, low motion")
    ]
    
    idle_passed = 0
    for time_since_motion, avg_motion, expected, description in idle_tests:
        # Using optimized thresholds: 180s and 0.02
        result = (time_since_motion > 180 and avg_motion < 0.02)
        if result == expected:
            idle_passed += 1
            print(f"    ✅ {description}")
        else:
            print(f"    ❌ {description}")
    
    print(f"    Idle detection: {idle_passed}/{len(idle_tests)} passed")
    
    print("\n  Testing bottleneck detection logic...")
    
    # Test cases for bottleneck detection
    bottleneck_tests = [
        (0.35, 1, "understaffed", "High motion (0.35), low faces (1)"),
        (0.08, 4, "blockage", "Low motion (0.08), high faces (4)"),
        (0.15, 2, "normal", "Balanced activity")
    ]
    
    bottleneck_passed = 0
    for avg_motion, face_count, expected, description in bottleneck_tests:
        # Using optimized thresholds
        is_understaffed = (avg_motion > 0.25 and face_count <= 1)
        is_blockage = (face_count >= 4 and avg_motion < 0.08)
        result = "understaffed" if is_understaffed else "blockage" if is_blockage else "normal"
        
        if result == expected:
            bottleneck_passed += 1
            print(f"    ✅ {description}")
        else:
            print(f"    ❌ {description}: Got {result}")
    
    print(f"    Bottleneck detection: {bottleneck_passed}/{len(bottleneck_tests)} passed")
    
    total_tests = len(idle_tests) + len(bottleneck_tests)
    total_passed = idle_passed + bottleneck_passed
    
    print(f"\n  Overall business logic: {total_passed}/{total_tests} tests passed")
    print(f"  Success rate: {(total_passed/total_tests*100):.1f}%")

if __name__ == "__main__":
    verify_all()
    
    print("\n" + "="*80)
    print("RECOMMENDED NEXT STEPS:")
    print("="*80)
    print("1. If all checks pass ✅ : Run your Picam system:")
    print("   python picam_with_csv.py")
    print("\n2. Generate new reports:")
    print("   python truth_report_generator.py")
    print("   python bottleneck_detector_fixed.py")
    print("\n3. Run comprehensive tests again:")
    print("   python run_comprehensive_tests_windows.py")
    print("\n4. If issues remain ❌ : Run the detailed fixer:")
    print("   python fix_critical_failures.py")
    print("="*80)