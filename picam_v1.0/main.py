#!/usr/bin/env python3
"""
PICAM v1.0 - Main Entry Point
Frozen Core Logic with Security & USB Deployment
"""

import os
import sys
import time
import hashlib
import getpass
from datetime import datetime
from pathlib import Path

# Add current directory to path for frozen EXE
if getattr(sys, 'frozen', False):
    sys.path.insert(0, sys._MEIPASS)

# Import security and core modules
from security import (
    LiveTestProtection,
    MachineBinding,
    SecurityConfig,
    validate_environment
)
from frozen_core import (
    FrozenVideoProcessor,
    FrozenIdleDetector,
    FrozenBottleneckAnalyzer,
    TruthReportGenerator,
    PICAM_CORE_VERSION
)
from license_manager import LicenseManager, LicenseTier

class PicamSystem:
    """Main Picam System - Entry point for frozen v1.0"""
    
    def __init__(self):
        self.version = "1.0.0"
        self.core_version = PICAM_CORE_VERSION
        self.security = None
        self.license_manager = None
        self.video_processor = None
        self.idle_detector = None
        self.bottleneck_analyzer = None
        self.report_generator = None
        self.is_running = False
        
        # Paths
        self.base_dir = Path(sys._MEIPASS if getattr(sys, 'frozen', False) else os.path.dirname(__file__))
        self.config_dir = self.base_dir / "configs"
        self.assets_dir = self.base_dir / "assets"
        self.reports_dir = Path("picam_reports")  # Will be created in working directory
        
    def initialize_security(self):
        """Initialize security subsystems."""
        print("\n" + "="*60)
        print("PICAM v1.0 - SECURITY INITIALIZATION")
        print("="*60)
        
        # Create security config
        security_config = SecurityConfig(
            require_live_test=True,
            require_machine_binding=True,
            allow_fallback=True,
            max_password_attempts=3,
            lockout_time_minutes=5
        )
        
        # Initialize security
        self.security = LiveTestProtection(security_config)
        
        # Load banner
        banner_file = self.assets_dir / "banner.txt"
        if banner_file.exists():
            with open(banner_file, 'r', encoding='utf-8') as f:
                print(f.read())
        
        # 1. Live Test Password
        if security_config.require_live_test:
            print("\n🔒 LIVE TEST PROTECTION")
            print("-" * 40)
            if not self.security.prompt_live_test():
                print("❌ Live test failed. Exiting.")
                return False
        
        # 2. Machine Binding
        if security_config.require_machine_binding:
            print("\n🖥️  MACHINE VALIDATION")
            print("-" * 40)
            machine_binding = MachineBinding()
            if not machine_binding.validate_current_machine():
                if not security_config.allow_fallback:
                    print("❌ Machine binding failed. Exiting.")
                    return False
                print("⚠️  Machine binding failed, but fallback allowed.")
        
        # 3. Environment Validation
        print("\n🔍 ENVIRONMENT VALIDATION")
        print("-" * 40)
        env_valid, env_issues = validate_environment()
        if not env_valid:
            print(f"⚠️  Environment issues found:")
            for issue in env_issues:
                print(f"   - {issue}")
            print("Proceeding with warnings...")
        
        return True
    
    def initialize_license(self):
        """Initialize license manager (future-ready)."""
        print("\n📋 LICENSE INITIALIZATION")
        print("-" * 40)
        
        self.license_manager = LicenseManager()
        tier = self.license_manager.get_current_tier()
        
        print(f"✓ License Tier: {tier.value}")
        print(f"✓ Features: {', '.join(self.license_manager.get_available_features())}")
        
        return True
    
    def initialize_core(self):
        """Initialize frozen core logic."""
        print("\n🚀 CORE SYSTEM INITIALIZATION")
        print("-" * 40)
        
        print(f"✓ Core Version: {self.core_version}")
        print(f"✓ Build Version: {self.version}")
        
        # Load zone configuration
        zones_config = self.config_dir / "zones_config.json"
        if not zones_config.exists():
            print(f"❌ Zones config not found: {zones_config}")
            # Create default config
            default_zones = [
                {
                    "name": "Counter",
                    "points": [[100, 100], [300, 100], [300, 300], [100, 300]],
                    "zone_type": "high_risk",
                    "idle_threshold": 180,
                    "motion_threshold": 0.02
                }
            ]
            import json
            zones_config.parent.mkdir(exist_ok=True)
            with open(zones_config, 'w') as f:
                json.dump(default_zones, f, indent=2)
            print("✓ Created default zones configuration")
        
        # Initialize frozen components
        self.video_processor = FrozenVideoProcessor()
        self.idle_detector = FrozenIdleDetector()
        self.bottleneck_analyzer = FrozenBottleneckAnalyzer()
        self.report_generator = TruthReportGenerator()
        
        # Create reports directory
        self.reports_dir.mkdir(exist_ok=True)
        
        print("✓ All core components initialized")
        print("✓ Reports directory: ", self.reports_dir.absolute())
        
        return True
    
    def run_monitoring_loop(self):
        """Main monitoring loop (simplified for example)."""
        print("\n🎯 STARTING MONITORING")
        print("-" * 40)
        print("Press Ctrl+C to stop monitoring")
        
        self.is_running = True
        frame_count = 0
        
        try:
            while self.is_running:
                frame_count += 1
                
                # Simulate frame processing
                frame_data = {
                    "frame_id": frame_count,
                    "timestamp": datetime.now().isoformat(),
                    "simulated_people": 2,
                    "simulated_motion": 0.15
                }
                
                # Process with frozen core logic
                detection_result = self.video_processor.process_frame(frame_data)
                idle_status = self.idle_detector.check_idle(detection_result)
                bottleneck_status = self.bottleneck_analyzer.analyze(detection_result)
                
                # Generate report every 10 frames
                if frame_count % 10 == 0:
                    report_data = {
                        "detection": detection_result,
                        "idle_status": idle_status,
                        "bottleneck_status": bottleneck_status,
                        "frame_count": frame_count
                    }
                    self.report_generator.generate_report(report_data, self.reports_dir)
                    print(f"✓ Frame {frame_count}: Report generated")
                
                # Simulate processing delay
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Monitoring stopped by user")
        except Exception as e:
            print(f"\n❌ Error in monitoring loop: {e}")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Clean shutdown."""
        print("\n🔴 SHUTTING DOWN")
        print("-" * 40)
        
        self.is_running = False
        
        # Generate final report
        if self.report_generator:
            final_report = {
                "shutdown_time": datetime.now().isoformat(),
                "message": "System shutdown"
            }
            self.report_generator.generate_report(final_report, self.reports_dir, "final_report.json")
            print("✓ Final report generated")
        
        print("✓ Picam v1.0 shutdown complete")
        print("=" * 60)
    
    def run(self):
        """Main execution flow."""
        try:
            # Step 1: Security
            if not self.initialize_security():
                return
            
            # Step 2: License (future-ready)
            if not self.initialize_license():
                return
            
            # Step 3: Core
            if not self.initialize_core():
                return
            
            # Step 4: Monitoring
            self.run_monitoring_loop()
            
        except Exception as e:
            print(f"\n❌ FATAL ERROR: {e}")
            import traceback
            traceback.print_exc()
            return 1
        
        return 0

# Frozen logic decorator
def frozen_logic(version="1.0.0"):
    """Decorator to mark functions as frozen logic."""
    def decorator(func):
        func._frozen_version = version
        func._frozen_hash = hashlib.sha256(func.__name__.encode()).hexdigest()[:8]
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Add version validation in production
            return func(*args, **kwargs)
        
        wrapper._is_frozen = True
        return wrapper
    return decorator

if __name__ == "__main__":
    # Entry point for both script and frozen EXE
    app = PicamSystem()
    sys.exit(app.run())