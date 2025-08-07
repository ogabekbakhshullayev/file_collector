"""
Advanced Android Device Fraud Detection System
Data Processing and Feature Engineering Module

This module handles the extraction and processing of discriminative features
from Android device JSON metadata for real vs emulator classification.
"""

import json
import re
from typing import Dict, Any, List, Tuple
from pathlib import Path

# Try to import numpy and pandas, use fallback if not available
try:
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    NUMPY_AVAILABLE = True
    PANDAS_AVAILABLE = True
    TQDM_AVAILABLE = True
except (ImportError, AttributeError, ModuleNotFoundError) as e:
    NUMPY_AVAILABLE = False
    PANDAS_AVAILABLE = False
    TQDM_AVAILABLE = False
    
    # Create simple fallbacks for numpy functions we use
    class NumpyFallback:
        @staticmethod
        def mean(values):
            return sum(values) / len(values) if values else 0
        
        @staticmethod
        def max(values):
            return max(values) if values else 0
        
        @staticmethod
        def sum(values):
            return sum(values) if values else 0
        
        @staticmethod
        def array(values):
            return list(values) if values else []
        
        @staticmethod
        def inf():
            return float('inf')
        
        @staticmethod
        def number():
            return (int, float)
    
    np = NumpyFallback()
    np.inf = float('inf')
    np.number = (int, float)
    
    # Simple pandas DataFrame fallback
    class DataFrameFallback:
        def __init__(self, data):
            self.data = data if isinstance(data, list) else [data]
        
        def select_dtypes(self, include=None):
            return self
        
        @property 
        def columns(self):
            if self.data:
                return list(self.data[0].keys()) if isinstance(self.data[0], dict) else []
            return []
    
    class PandasFallback:
        @staticmethod
        def DataFrame(data):
            return DataFrameFallback(data)
    
    pd = PandasFallback()
    
    # Simple tqdm fallback
    def tqdm(iterable, *args, **kwargs):
        return iterable

import warnings
warnings.filterwarnings('ignore')


class AndroidDeviceFeatureExtractor:
    """
    Advanced feature extractor for Android device metadata analysis.
    Focuses on hardware, system properties, and behavioral patterns that
    distinguish real devices from emulators.
    """
    
    def __init__(self):
        self.features = []
        self.emulator_signatures = self._load_emulator_signatures()
    
    def _load_emulator_signatures(self) -> Dict[str, List[str]]:
        """Known emulator signatures and patterns"""
        return {
            'hardware_patterns': [
                'android_x86', 'goldfish', 'ranchu', 'vbox', 'virtualbox',
                'qemu', 'sdk_gphone', 'generic', 'vsoc'
            ],
            'build_patterns': [
                'generic', 'sdk', 'aosp', 'test-keys', 'google/sdk'
            ],
            'cpu_patterns': [
                'intel', 'genuineintel', 'x86', 'i686', 'virtualbox'
            ],
            'suspicious_features': [
                'android.hardware.telephony.cdma', 
                'android.software.leanback',
                'com.google.android.feature.GOOGLE_BUILD'
            ]
        }
    
    def extract_camera_features(self, cameras: List[Dict]) -> Dict[str, Any]:
        """Extract discriminative camera-related features"""
        features = {}
        
        if not cameras:
            features.update({
                'num_cameras': 0,
                'has_back_camera': False,
                'has_front_camera': False,
                'max_resolution_width': 0,
                'max_resolution_height': 0,
                'avg_resolution_ratio': 0.0,
                'unique_formats': 0,
                'resolution_variety_score': 0.0
            })
            return features
        
        # Basic camera counts
        features['num_cameras'] = len(cameras)
        features['has_back_camera'] = any(cam.get('facing') == 'Back' for cam in cameras)
        features['has_front_camera'] = any(cam.get('facing') == 'Front' for cam in cameras)
        
        # Resolution analysis
        all_resolutions = []
        all_formats = set()
        
        for camera in cameras:
            capabilities = camera.get('capabilities', {})
            resolutions = capabilities.get('resolutions', [])
            
            for res in resolutions:
                width = res.get('width', 0)
                height = res.get('height', 0)
                format_str = res.get('formatString', '')
                
                if width > 0 and height > 0:
                    all_resolutions.append((width, height))
                    all_formats.add(format_str)
        
        if all_resolutions:
            widths, heights = zip(*all_resolutions)
            features['max_resolution_width'] = max(widths)
            features['max_resolution_height'] = max(heights)
            
            # Calculate average aspect ratio
            ratios = [w/h if h > 0 else 0 for w, h in all_resolutions]
            features['avg_resolution_ratio'] = np.mean(ratios) if ratios else 0.0
            
            # Resolution variety score (higher = more diverse resolutions)
            unique_resolutions = len(set(all_resolutions))
            features['resolution_variety_score'] = unique_resolutions / len(all_resolutions)
        else:
            features.update({
                'max_resolution_width': 0,
                'max_resolution_height': 0,
                'avg_resolution_ratio': 0.0,
                'resolution_variety_score': 0.0
            })
        
        features['unique_formats'] = len(all_formats)
        
        # Emulator-specific camera patterns
        features['has_low_res_only'] = features['max_resolution_width'] < 1920
        features['has_standard_emulator_res'] = any(
            (w, h) in [(176, 144), (320, 240), (640, 480), (1280, 720)]
            for w, h in all_resolutions
        )
        
        return features
    
    def extract_cpu_features(self, cpuinfo: List[Dict]) -> Dict[str, Any]:
        """Extract CPU and hardware features"""
        features = {}
        
        if not cpuinfo:
            return {
                'num_cpu_cores': 0,
                'is_intel_cpu': False,
                'is_arm_cpu': False,
                'cpu_arch_score': 0.0,
                'has_virtualbox_cpu': False,
                'cpu_model_consistency': 0.0
            }
        
        features['num_cpu_cores'] = len(cpuinfo)
        
        # Analyze CPU information
        cpu_models = []
        cpu_vendors = []
        cpu_architectures = []
        
        for cpu in cpuinfo:
            fields = cpu.get('fields', {})
            model_name = fields.get('model name', '').lower()
            vendor_id = fields.get('vendor_id', '').lower()
            cpu_arch = fields.get('CPU architecture', '')
            
            cpu_models.append(model_name)
            cpu_vendors.append(vendor_id)
            cpu_architectures.append(cpu_arch)
        
        # CPU vendor analysis
        intel_indicators = ['intel', 'genuineintel']
        arm_indicators = ['armv', 'arm']
        
        features['is_intel_cpu'] = any(
            any(indicator in vendor for indicator in intel_indicators)
            for vendor in cpu_vendors if vendor
        )
        features['is_arm_cpu'] = any(
            any(indicator in model for indicator in arm_indicators)
            for model in cpu_models if model
        )
        
        # VirtualBox detection
        features['has_virtualbox_cpu'] = any(
            'virtualbox' in model or 'vbox' in model
            for model in cpu_models if model
        )
        
        # CPU consistency score (real devices have consistent CPU info)
        unique_models = len(set(cpu_models))
        features['cpu_model_consistency'] = 1.0 if unique_models == 1 else 0.0
        
        # Architecture scoring (ARM more likely real, x86 more likely emulator)
        if features['is_arm_cpu']:
            features['cpu_arch_score'] = 1.0  # Real device likely
        elif features['is_intel_cpu']:
            features['cpu_arch_score'] = -1.0  # Emulator likely
        else:
            features['cpu_arch_score'] = 0.0
        
        return features
    
    def extract_system_features(self, getprop: List[Dict]) -> Dict[str, Any]:
        """Extract system properties features"""
        features = {}
        
        if not getprop:
            return {
                'build_type_score': 0.0,
                'is_debug_build': False,
                'has_test_keys': False,
                'hardware_emulator_score': 0.0,
                'build_fingerprint_score': 0.0,
                'ro_secure': 0,
                'ro_debuggable': 0
            }
        
        # Convert getprop list to dictionary for easier access
        props = {prop.get('key', ''): prop.get('value', '') for prop in getprop}
        
        # Build type analysis
        build_type = props.get('ro.build.type', '').lower()
        build_tags = props.get('ro.build.tags', '').lower()
        build_flavor = props.get('ro.build.flavor', '').lower()
        
        features['is_debug_build'] = build_type in ['userdebug', 'eng']
        features['has_test_keys'] = 'test-keys' in build_tags
        
        # Hardware detection
        hardware = props.get('ro.hardware', '').lower()
        product_board = props.get('ro.product.board', '').lower()
        product_device = props.get('ro.product.device', '').lower()
        
        # Score based on emulator hardware patterns
        emulator_score = 0.0
        for pattern in self.emulator_signatures['hardware_patterns']:
            if pattern in hardware or pattern in product_board or pattern in product_device:
                emulator_score += 1.0
        
        features['hardware_emulator_score'] = emulator_score
        
        # Build fingerprint analysis
        fingerprint = props.get('ro.build.fingerprint', '').lower()
        fingerprint_score = 0.0
        for pattern in self.emulator_signatures['build_patterns']:
            if pattern in fingerprint:
                fingerprint_score += 1.0
        features['build_fingerprint_score'] = fingerprint_score
        
        # Security properties
        features['ro_secure'] = int(props.get('ro.secure', '0'))
        features['ro_debuggable'] = int(props.get('ro.debuggable', '0'))
        
        # Build type scoring
        if build_type == 'user':
            features['build_type_score'] = 1.0  # More likely real
        elif build_type in ['userdebug', 'eng']:
            features['build_type_score'] = -1.0  # More likely emulator
        else:
            features['build_type_score'] = 0.0
        
        return features
    
    def extract_hardware_features(self, features_list: List[Dict], 
                                  uname: Dict) -> Dict[str, Any]:
        """Extract hardware and system features"""
        feature_dict = {}
        
        # Extract feature names - handle both dict and string formats
        feature_names = []
        for f in features_list:
            if isinstance(f, dict):
                feature_names.append(f.get('name', ''))
            elif isinstance(f, str):
                feature_names.append(f)
            else:
                # Skip non-dict, non-string items
                continue
        
        # Count important feature categories
        feature_dict['num_hardware_features'] = len(feature_names)
        
        # Hardware capabilities
        camera_features = [f for f in feature_names if 'camera' in f.lower()]
        sensor_features = [f for f in feature_names if 'sensor' in f.lower()]
        telephony_features = [f for f in feature_names if 'telephony' in f.lower()]
        bluetooth_features = [f for f in feature_names if 'bluetooth' in f.lower()]
        
        feature_dict['num_camera_features'] = len(camera_features)
        feature_dict['num_sensor_features'] = len(sensor_features)
        feature_dict['num_telephony_features'] = len(telephony_features)
        feature_dict['num_bluetooth_features'] = len(bluetooth_features)
        
        # Emulator-specific feature detection
        google_features = [f for f in feature_names if 'google' in f.lower()]
        samsung_features = [f for f in feature_names if 'samsung' in f.lower()]
        
        feature_dict['num_google_features'] = len(google_features)
        feature_dict['num_samsung_features'] = len(samsung_features)
        
        # System architecture from uname
        if uname:
            machine = uname.get('machine', '').lower()
            sysname = uname.get('sysname', '').lower()
            release = uname.get('release', '').lower()
            
            feature_dict['is_x86_arch'] = 'i686' in machine or 'x86' in machine
            feature_dict['is_arm_arch'] = 'arm' in machine
            feature_dict['is_linux_kernel'] = sysname == 'linux'
            
            # Kernel version analysis (emulators often have specific patterns)
            feature_dict['kernel_version_score'] = self._analyze_kernel_version(release)
        else:
            feature_dict.update({
                'is_x86_arch': False,
                'is_arm_arch': False,
                'is_linux_kernel': False,
                'kernel_version_score': 0.0
            })
        
        return feature_dict
    
    def _analyze_kernel_version(self, release: str) -> float:
        """Analyze kernel version for emulator patterns"""
        # Common emulator kernel patterns
        emulator_patterns = ['generic', 'qemu', 'goldfish', 'ranchu']
        
        score = 0.0
        for pattern in emulator_patterns:
            if pattern in release:
                score -= 1.0  # Negative score for emulator patterns
        
        # Real device kernels often have OEM-specific identifiers
        oem_patterns = ['samsung', 'qualcomm', 'mtk', 'exynos', 'snapdragon']
        for pattern in oem_patterns:
            if pattern in release:
                score += 1.0  # Positive score for OEM patterns
        
        return score
    
    def extract_package_features(self, packages: List[Dict]) -> Dict[str, Any]:
        """Extract package and app-related features"""
        features = {}
        
        if not packages:
            return {
                'num_installed_packages': 0,
                'google_apps_ratio': 0.0,
                'system_apps_ratio': 0.0,
                'emulator_app_score': 0.0
            }
        
        # Extract package names - handle both dict and string formats
        package_names = []
        for pkg in packages:
            if isinstance(pkg, dict):
                package_names.append(pkg.get('name', ''))
            elif isinstance(pkg, str):
                package_names.append(pkg)
            else:
                continue
                
        features['num_installed_packages'] = len(package_names)
        
        # Google apps analysis
        google_apps = [p for p in package_names if 'google' in p.lower()]
        features['google_apps_ratio'] = len(google_apps) / len(package_names) if package_names else 0.0
        
        # System apps analysis
        system_apps = [p for p in package_names if p.startswith('com.android.')]
        features['system_apps_ratio'] = len(system_apps) / len(package_names) if package_names else 0.0
        
        # Emulator-specific app detection
        emulator_apps = [
            'com.android.emulator',
            'com.google.android.apps.chromecast.app',
            'com.android.development',
            'com.android.cts'
        ]
        
        emulator_score = sum(1 for app in emulator_apps if app in package_names)
        features['emulator_app_score'] = emulator_score
        
        return features
    
    def extract_network_features(self, mount: List[Dict]) -> Dict[str, Any]:
        """Extract network and filesystem features"""
        features = {}
        
        if not mount:
            return {
                'num_mount_points': 0,
                'has_vbox_mounts': False,
                'has_emulator_fs': False,
                'virtual_fs_score': 0.0
            }
        
        mount_points = [m.get('mountPoint', '') for m in mount]
        devices = [m.get('device', '') for m in mount]
        fs_types = [m.get('type', '') for m in mount]
        
        features['num_mount_points'] = len(mount_points)
        
        # VirtualBox detection
        vbox_indicators = ['vbox', 'virtualbox', 'Applications', 'Pictures', 'Misc']
        features['has_vbox_mounts'] = any(
            any(indicator in device.lower() for indicator in vbox_indicators)
            for device in devices
        )
        
        # Emulator filesystem detection
        emulator_fs = ['tmpfs', 'debugfs', 'configfs']
        virtual_score = sum(1 for fs in fs_types if fs in emulator_fs)
        features['virtual_fs_score'] = virtual_score / len(fs_types) if fs_types else 0.0
        
        # Emulator-specific mount points
        emulator_mounts = ['/mnt/shared', '/system/lib/arm']
        features['has_emulator_fs'] = any(
            any(em_mount in mp for em_mount in emulator_mounts)
            for mp in mount_points
        )
        
        return features
    
    def extract_all_features(self, json_data: Dict) -> Dict[str, Any]:
        """Extract all features from a single JSON file"""
        features = {}
        
        # Extract features from different sections
        cameras = json_data.get('cameras', [])
        cpuinfo = json_data.get('cpuinfo', [])
        getprop = json_data.get('getprop', [])
        hardware_features = json_data.get('features', [])
        uname = json_data.get('uname', {})
        packages = json_data.get('packages', [])
        mount = json_data.get('mount', [])
        
        # Combine all feature extractors
        features.update(self.extract_camera_features(cameras))
        features.update(self.extract_cpu_features(cpuinfo))
        features.update(self.extract_system_features(getprop))
        features.update(self.extract_hardware_features(hardware_features, uname))
        features.update(self.extract_package_features(packages))
        features.update(self.extract_network_features(mount))
        
        # Calculate composite scores
        features['emulator_likelihood_score'] = self._calculate_emulator_score(features)
        
        return features
    
    def _calculate_emulator_score(self, features: Dict[str, Any]) -> float:
        """Calculate composite emulator likelihood score"""
        score = 0.0
        
        # Hardware indicators
        if features.get('is_intel_cpu', False):
            score += 2.0
        if features.get('is_arm_cpu', False):
            score -= 1.0
        
        # System indicators
        score += features.get('hardware_emulator_score', 0.0)
        score += features.get('build_fingerprint_score', 0.0)
        score += features.get('build_type_score', 0.0)
        
        # Architecture indicators
        if features.get('is_x86_arch', False):
            score += 1.5
        if features.get('is_arm_arch', False):
            score -= 1.0
        
        # Mount/filesystem indicators
        if features.get('has_vbox_mounts', False):
            score += 2.0
        if features.get('has_emulator_fs', False):
            score += 1.0
        
        score += features.get('virtual_fs_score', 0.0)
        score += features.get('emulator_app_score', 0.0)
        
        return score


class DataProcessor:
    """Main data processing class for the fraud detection system"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.feature_extractor = AndroidDeviceFeatureExtractor()
        
    def load_and_process_data(self) -> Tuple[Any, Any]:
        """Load and process all JSON files to create training dataset"""
        
        real_path = self.data_path / "real"
        emulator_path = self.data_path / "emulator"
        
        all_features = []
        labels = []
        
        print("Processing real device data...")
        # Process real devices
        for json_file in tqdm(real_path.glob("*.json")):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                features = self.feature_extractor.extract_all_features(data)
                features['filename'] = json_file.name
                features['device_type'] = 'real'
                
                all_features.append(features)
                labels.append(0)  # 0 for real devices
                
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
                continue
        
        print("Processing emulator data...")
        # Process emulators
        for json_file in tqdm(emulator_path.glob("*.json")):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                features = self.feature_extractor.extract_all_features(data)
                features['filename'] = json_file.name
                features['device_type'] = 'emulator'
                
                all_features.append(features)
                labels.append(1)  # 1 for emulators
                
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
                continue
        
        # Create DataFrame
        df = pd.DataFrame(all_features)
        if NUMPY_AVAILABLE:
            labels = np.array(labels)
        
        print(f"Processed {len(all_features)} devices")
        if hasattr(df, 'data') and df.data:  # For fallback DataFrame
            real_count = sum(1 for item in df.data if item.get('device_type') == 'real')
            emulator_count = sum(1 for item in df.data if item.get('device_type') == 'emulator')
        else:  # For real pandas DataFrame
            real_count = len(df[df['device_type'] == 'real']) if PANDAS_AVAILABLE else 0
            emulator_count = len(df[df['device_type'] == 'emulator']) if PANDAS_AVAILABLE else 0
            
        print(f"Real devices: {real_count}, Emulators: {emulator_count}")
        
        return df, labels
    
    def prepare_features(self, df: Any) -> Any:
        """Prepare features for machine learning"""
        
        if not PANDAS_AVAILABLE:
            # Fallback for when pandas is not available
            if hasattr(df, 'data') and df.data:
                # Extract numeric features from the fallback DataFrame
                numeric_features = {}
                for item in df.data:
                    for key, value in item.items():
                        if key not in ['filename', 'device_type'] and isinstance(value, (int, float)):
                            if key not in numeric_features:
                                numeric_features[key] = []
                            numeric_features[key].append(value)
                
                # Fill missing values with 0
                max_len = max(len(values) for values in numeric_features.values()) if numeric_features else 0
                for key in numeric_features:
                    while len(numeric_features[key]) < max_len:
                        numeric_features[key].append(0)
                
                print(f"Prepared {len(numeric_features)} features for training (fallback mode)")
                print("Feature columns:", list(numeric_features.keys())[:10])  # Show first 10
                
                return numeric_features
        
        # Remove non-numeric columns
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove filename and device_type if they exist
        feature_cols = [col for col in feature_cols if col not in ['filename', 'device_type']]
        
        X = df[feature_cols].copy()
        
        # Handle missing values
        X = X.fillna(0)
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], 0)
        
        print(f"Prepared {len(feature_cols)} features for training")
        print("Feature columns:", feature_cols[:10])  # Show first 10
        
        return X


if __name__ == "__main__":
    # Example usage
    processor = DataProcessor("/Users/bnutfilloyev/Developer/AIGroup/fraud-android")
    df, labels = processor.load_and_process_data()
    X = processor.prepare_features(df)
    
    if NUMPY_AVAILABLE:
        print(f"Dataset shape: {X.shape if hasattr(X, 'shape') else 'N/A'}")
        print(f"Label distribution: Real={np.sum(labels==0)}, Emulator={np.sum(labels==1)}")
    else:
        print("Dataset processed using fallback mode (numpy not available)")
        print(f"Features extracted: {len(X) if isinstance(X, dict) else 'N/A'}")
