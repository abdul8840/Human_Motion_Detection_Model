"""
LICENSE MANAGER - Future-ready license system
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Set
from enum import Enum
from dataclasses import dataclass
import datetime

class LicenseTier(Enum):
    """License tiers."""
    FREE = "FREE"
    PRO = "PRO"
    PREMIUM = "PREMIUM"
    ENTERPRISE = "ENTERPRISE"

@dataclass
class License:
    """License data structure."""
    tier: LicenseTier
    customer_id: str
    expiry_date: Optional[datetime.datetime]
    features: List[str]
    signature: str  # Digital signature for validation

class LicenseManager:
    """License manager with feature gating."""
    
    # Feature definitions per tier
    TIER_FEATURES = {
        LicenseTier.FREE: {
            "core_detection",
            "basic_reports",
            "live_test",
            "single_camera"
        },
        LicenseTier.PRO: {
            "core_detection",
            "basic_reports",
            "live_test",
            "multi_camera",
            "advanced_analytics",
            "predictive_bottlenecks",
            "custom_zones"
        },
        LicenseTier.PREMIUM: {
            "core_detection",
            "basic_reports",
            "live_test",
            "multi_camera",
            "advanced_analytics",
            "predictive_bottlenecks",
            "custom_zones",
            "realtime_optimization",
            "enterprise_integration",
            "custom_ai_models",
            "priority_support"
        }
    }
    
    def __init__(self, license_file: str = "picam_license.lic"):
        self.license_file = Path(license_file)
        self.current_license: Optional[License] = None
        self.current_tier = LicenseTier.FREE
        
        # Try to load license
        self._load_license()
    
    def _load_license(self):
        """Load license from file."""
        if not self.license_file.exists():
            print("⚠️  No license file found. Using FREE tier.")
            return
        
        try:
            with open(self.license_file, 'r') as f:
                data = json.load(f)
            
            # Validate license
            if self._validate_license(data):
                self.current_license = License(
                    tier=LicenseTier(data['tier']),
                    customer_id=data['customer_id'],
                    expiry_date=datetime.datetime.fromisoformat(data['expiry']) if data.get('expiry') else None,
                    features=data['features'],
                    signature=data['signature']
                )
                self.current_tier = self.current_license.tier
                print(f"✓ License loaded: {self.current_tier.value}")
            else:
                print("⚠️  License validation failed. Using FREE tier.")
                
        except Exception as e:
            print(f"⚠️  Error loading license: {e}. Using FREE tier.")
    
    def _validate_license(self, license_data: Dict) -> bool:
        """Validate license signature."""
        # Simplified validation - in production would use proper crypto
        required_keys = {'tier', 'customer_id', 'features', 'signature'}
        if not all(k in license_data for k in required_keys):
            return False
        
        # Check expiry
        if license_data.get('expiry'):
            expiry = datetime.datetime.fromisoformat(license_data['expiry'])
            if expiry < datetime.datetime.now():
                print(f"⚠️  License expired on {expiry.date()}")
                return False
        
        return True
    
    def get_current_tier(self) -> LicenseTier:
        """Get current license tier."""
        return self.current_tier
    
    def get_available_features(self) -> List[str]:
        """Get available features for current tier."""
        features = self.TIER_FEATURES.get(self.current_tier, set())
        
        # Add any additional features from license file
        if self.current_license:
            features.update(self.current_license.features)
        
        return sorted(features)
    
    def is_feature_available(self, feature: str) -> bool:
        """Check if a feature is available in current tier."""
        return feature in self.get_available_features()
    
    def require_feature(self, feature: str):
        """Decorator to require a specific feature."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                if not self.is_feature_available(feature):
                    raise PermissionError(
                        f"Feature '{feature}' requires {LicenseTier.PRO.value}+ license. "
                        f"Current tier: {self.current_tier.value}"
                    )
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def upgrade_license(self, new_license_data: Dict):
        """Upgrade license (admin function)."""
        if self._validate_license(new_license_data):
            with open(self.license_file, 'w') as f:
                json.dump(new_license_data, f, indent=2)
            
            # Reload license
            self._load_license()
            print(f"✓ License upgraded to {self.current_tier.value}")
            return True
        else:
            print("❌ Invalid license data")
            return False
    
    def get_license_info(self) -> Dict:
        """Get license information for display."""
        if not self.current_license:
            return {
                "tier": "FREE",
                "expiry": None,
                "features": list(self.TIER_FEATURES[LicenseTier.FREE])
            }
        
        return {
            "tier": self.current_license.tier.value,
            "customer_id": self.current_license.customer_id,
            "expiry": self.current_license.expiry_date.isoformat() if self.current_license.expiry_date else None,
            "features": self.current_license.features
        }