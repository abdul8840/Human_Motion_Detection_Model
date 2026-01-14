"""
SECURITY MODULE - Live Test Protection & Machine Binding
"""

import os
import sys
import hashlib
import getpass
import socket
import uuid
import json
import platform
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings

@dataclass
class SecurityConfig:
    """Security configuration."""
    require_live_test: bool = True
    require_machine_binding: bool = True
    allow_fallback: bool = True
    max_password_attempts: int = 3
    lockout_time_minutes: int = 5
    password_hash: str = ""  # Will be set during build

class LiveTestProtection:
    """Live test password protection."""
    
    # Built-in password (hashed) - Set during build process
    # Default: "PicamLiveTest2024!"
    DEFAULT_PASSWORD_HASH = "a6f5f6d7e8c9b0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6"
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.attempts = 0
        self.locked_until = None
        
        # Load password hash from config or use default
        self.password_hash = config.password_hash or self.DEFAULT_PASSWORD_HASH
        
        # Lock file for tracking attempts
        self.lock_file = Path("picam_security.lock")
    
    def _hash_password(self, password: str) -> str:
        """Hash password with salt."""
        salt = "PICAM_v1.0_SALT_2024"
        return hashlib.sha256((password + salt).encode()).hexdigest()
    
    def _check_lockout(self) -> bool:
        """Check if system is locked out."""
        if not self.lock_file.exists():
            return False
        
        try:
            with open(self.lock_file, 'r') as f:
                lock_data = json.load(f)
            
            locked_until = datetime.fromisoformat(lock_data['locked_until'])
            if datetime.now() < locked_until:
                self.locked_until = locked_until
                return True
            else:
                # Lock expired, remove file
                self.lock_file.unlink(missing_ok=True)
                return False
                
        except Exception:
            return False
    
    def _set_lockout(self):
        """Set lockout period."""
        locked_until = datetime.now() + timedelta(minutes=self.config.lockout_time_minutes)
        self.locked_until = locked_until
        
        lock_data = {
            'locked_until': locked_until.isoformat(),
            'attempts': self.attempts,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.lock_file, 'w') as f:
            json.dump(lock_data, f)
    
    def prompt_live_test(self) -> bool:
        """Prompt for live test password."""
        if not self.config.require_live_test:
            return True
        
        # Check lockout
        if self._check_lockout():
            remaining = (self.locked_until - datetime.now()).seconds // 60
            print(f"🔒 System locked. Try again in {remaining} minutes.")
            return False
        
        print("\nLIVE TEST REQUIRED")
        print("-" * 40)
        print("Enter live test password to continue:")
        
        while self.attempts < self.config.max_password_attempts:
            try:
                # Use getpass to hide input
                password = getpass.getpass(f"Attempt {self.attempts + 1}/{self.config.max_password_attempts}: ")
                
                if self._hash_password(password) == self.password_hash:
                    print("✓ Live test passed!")
                    return True
                else:
                    self.attempts += 1
                    print(f"❌ Incorrect password. Attempts remaining: {self.config.max_password_attempts - self.attempts}")
                    
            except KeyboardInterrupt:
                print("\n⚠️  Live test cancelled.")
                return False
            except Exception as e:
                print(f"⚠️  Error: {e}")
                self.attempts += 1
        
        # Max attempts reached
        print(f"\n❌ Maximum attempts reached. System locked for {self.config.lockout_time_minutes} minutes.")
        self._set_lockout()
        return False

class MachineBinding:
    """Machine binding to prevent copying."""
    
    def __init__(self):
        self.machine_id_file = Path("picam_machine.id")
        self.fallback_allowed = True
    
    def _get_machine_fingerprint(self) -> Dict[str, str]:
        """Generate machine fingerprint."""
        fingerprint = {}
        
        try:
            # 1. MAC Address
            mac = ':'.join(['{:02x}'.format((uuid.getnode() >> elements) & 0xff) 
                           for elements in range(0, 8*6, 8)][::-1])
            fingerprint["mac"] = mac
        except:
            fingerprint["mac"] = "unknown"
        
        try:
            # 2. Hostname
            fingerprint["hostname"] = socket.gethostname()
        except:
            fingerprint["hostname"] = "unknown"
        
        try:
            # 3. Platform info
            fingerprint["platform"] = platform.platform()
            fingerprint["processor"] = platform.processor()
        except:
            fingerprint["platform"] = "unknown"
            fingerprint["processor"] = "unknown"
        
        try:
            # 4. Disk serial (Windows)
            if platform.system() == "Windows":
                result = subprocess.run(
                    ["wmic", "diskdrive", "get", "serialnumber"],
                    capture_output=True,
                    text=True,
                    shell=True
                )
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    fingerprint["disk_serial"] = lines[1].strip()
        except:
            fingerprint["disk_serial"] = "unknown"
        
        # Generate hash from fingerprint
        fingerprint_str = json.dumps(fingerprint, sort_keys=True)
        fingerprint["hash"] = hashlib.sha256(fingerprint_str.encode()).hexdigest()[:32]
        
        return fingerprint
    
    def _save_machine_id(self, fingerprint: Dict):
        """Save machine fingerprint."""
        with open(self.machine_id_file, 'w') as f:
            json.dump(fingerprint, f, indent=2)
    
    def _load_machine_id(self) -> Optional[Dict]:
        """Load saved machine fingerprint."""
        if not self.machine_id_file.exists():
            return None
        
        try:
            with open(self.machine_id_file, 'r') as f:
                return json.load(f)
        except:
            return None
    
    def validate_current_machine(self) -> bool:
        """Validate current machine matches stored fingerprint."""
        print("Validating machine binding...")
        
        current_fingerprint = self._get_machine_fingerprint()
        saved_fingerprint = self._load_machine_id()
        
        if saved_fingerprint is None:
            # First run on this machine
            print("✓ First run on this machine. Creating machine ID.")
            self._save_machine_id(current_fingerprint)
            print(f"  Machine ID: {current_fingerprint['hash'][:16]}...")
            return True
        
        # Compare fingerprints
        current_hash = current_fingerprint["hash"]
        saved_hash = saved_fingerprint.get("hash", "")
        
        if current_hash == saved_hash:
            print(f"✓ Machine validated: {current_hash[:16]}...")
            return True
        else:
            print(f"❌ Machine mismatch!")
            print(f"  Expected: {saved_hash[:16]}...")
            print(f"  Current:  {current_hash[:16]}...")
            
            # Show differences
            for key in set(current_fingerprint.keys()) | set(saved_fingerprint.keys()):
                if key == "hash":
                    continue
                current_val = current_fingerprint.get(key, "missing")
                saved_val = saved_fingerprint.get(key, "missing")
                if current_val != saved_val:
                    print(f"    {key}: {saved_val} → {current_val}")
            
            return False
    
    def force_bind_to_current(self):
        """Force bind to current machine (admin function)."""
        fingerprint = self._get_machine_fingerprint()
        self._save_machine_id(fingerprint)
        print(f"✓ Force-bound to current machine: {fingerprint['hash'][:16]}...")

def validate_environment() -> Tuple[bool, List[str]]:
    """Validate runtime environment."""
    issues = []
    
    # Check Python version
    if sys.version_info < (3, 10):
        issues.append(f"Python 3.10+ required (found {sys.version})")
    
    # Check write permissions
    test_file = Path("picam_test_write.tmp")
    try:
        test_file.write_text("test")
        test_file.unlink()
    except Exception as e:
        issues.append(f"No write permission: {e}")
    
    # Check disk space (at least 100MB free)
    try:
        import shutil
        free_gb = shutil.disk_usage(".").free / (1024**3)
        if free_gb < 0.1:  # 100MB
            issues.append(f"Low disk space: {free_gb:.1f}GB free")
    except:
        pass
    
    return len(issues) == 0, issues