"""
Demo script for Android Fraud Detection

This script demonstrates the fraud detection functionality with sample data.
"""

import json
import sys
from pathlib import Path

# Add the parent directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent))

from fraud_detection.predictor import ProductionAPI

def create_real_device_sample():
    """Create a sample real device metadata"""
    return {
        "device_model": "SM-G975F",
        "manufacturer": "samsung",
        "device_brand": "samsung",
        "android_version": "10",
        "api_level": 29,
        "screen_width": 1440,
        "screen_height": 3040,
        "screen_density": 550,
        "cpu_info": {
            "cores": 8,
            "frequency": 2730000,
            "architecture": "arm64-v8a"
        },
        "total_memory": 8589934592,
        "available_memory": 6442450944,
        "internal_storage": 128000000000,
        "external_storage": 0,
        "system_features": [
            "android.hardware.camera",
            "android.hardware.camera.autofocus",
            "android.hardware.camera.flash",
            "android.hardware.location.gps",
            "android.hardware.sensor.accelerometer",
            "android.hardware.sensor.gyroscope",
            "android.hardware.telephony",
            "android.hardware.wifi",
            "android.hardware.bluetooth"
        ],
        "build_info": {
            "type": "user",
            "tags": "release-keys",
            "fingerprint": "samsung/beyond2ltexx/beyond2:10/QP1A.190711.020/G975FXXU3ASL2:user/release-keys",
            "time": 1577836800000,
            "bootloader": "G975FXXU3ASL2",
            "radio": "G975FXXU3ASL2"
        },
        "cameras": [
            {
                "facing": "back",
                "max_width": 4032,
                "max_height": 3024,
                "has_flash": True,
                "has_autofocus": True
            },
            {
                "facing": "front",
                "max_width": 3264,
                "max_height": 2448,
                "has_flash": False,
                "has_autofocus": True
            },
            {
                "facing": "back",
                "max_width": 4032,
                "max_height": 3024,
                "has_flash": True,
                "has_autofocus": True
            }
        ],
        "sensors": [
            {"type": "accelerometer"},
            {"type": "gyroscope"},
            {"type": "magnetometer"},
            {"type": "proximity"},
            {"type": "light"},
            {"type": "pressure"},
            {"type": "temperature"},
            {"type": "humidity"}
        ],
        "installed_apps": [
            {"package_name": "com.android.chrome", "is_system": False},
            {"package_name": "com.samsung.android.contacts", "is_system": True},
            {"package_name": "com.samsung.android.messaging", "is_system": True},
            {"package_name": "com.whatsapp", "is_system": False}
        ]
    }

def create_emulator_sample():
    """Create a sample emulator metadata"""
    return {
        "device_model": "Android SDK built for x86_64",
        "manufacturer": "Google",
        "device_brand": "generic",
        "android_version": "10",
        "api_level": 29,
        "screen_width": 1080,
        "screen_height": 1920,
        "screen_density": 420,
        "cpu_info": {
            "cores": 4,
            "frequency": 2800000,
            "architecture": "x86_64"
        },
        "total_memory": 2147483648,
        "available_memory": 1073741824,
        "internal_storage": 32000000000,
        "external_storage": 0,
        "system_features": [
            "android.hardware.camera",
            "android.hardware.location.gps"
        ],
        "build_info": {
            "type": "userdebug",
            "tags": "test-keys",
            "fingerprint": "google/sdk_gphone_x86_64/generic_x86_64:10/QSR1.190920.001/5891938:userdebug/test-keys",
            "time": 1577836800000
        },
        "cameras": [
            {
                "facing": "back",
                "max_width": 1920,
                "max_height": 1080,
                "has_flash": False,
                "has_autofocus": False
            }
        ],
        "sensors": [
            {"type": "accelerometer"},
            {"type": "gyroscope"}
        ],
        "installed_apps": [
            {"package_name": "com.android.calculator2", "is_system": True},
            {"package_name": "com.android.calendar", "is_system": True}
        ]
    }

def print_analysis_result(device_type, result):
    """Print formatted analysis result"""
    print(f"\n{'='*60}")
    print(f"ANALYSIS RESULT - {device_type.upper()}")
    print(f"{'='*60}")
    
    if result['status'] == 'error':
        print(f"❌ Error: {result.get('error_message', 'Unknown error')}")
        return
    
    analysis = result['result']
    
    # Prediction
    prediction = "🚨 EMULATOR" if analysis['is_emulator'] else "✅ REAL DEVICE"
    print(f"Prediction: {prediction}")
    print(f"Confidence Score: {analysis['confidence_score']:.3f}")
    print(f"Risk Level: {analysis['risk_level'].upper()}")
    print(f"Processing Time: {analysis['processing_time_ms']} ms")
    
    # Hardware Analysis
    summary = analysis['analysis_summary']
    print(f"\n📱 Hardware Analysis:")
    print(f"  CPU Architecture: {summary['cpu_architecture']}")
    print(f"  Cameras Detected: {summary['cameras_detected']}")
    print(f"  Risk Factors: {summary['risk_factors_count']}")
    print(f"  Confidence Factors: {summary['confidence_factors_count']}")
    
    # Metadata
    metadata = result['metadata']
    print(f"\n📊 Analysis Metadata:")
    print(f"  Features Analyzed: {metadata['features_analyzed']}")
    print(f"  Model Version: {analysis['model_version']}")
    print(f"  Timestamp: {metadata['timestamp']}")

def demo_single_predictions():
    """Demonstrate single device predictions"""
    print("🔍 ANDROID FRAUD DETECTION DEMO")
    print("Initializing fraud detection API...")
    
    # Initialize the API
    api = ProductionAPI(model_name="xgboost")
    
    # Show available models
    print("\n📋 Available Models:")
    available = api.get_available_models()
    for model in available.get('available_models', []):
        current = " (current)" if model == available.get('current_model') else ""
        print(f"  - {model}{current}")
    
    # Test real device
    print("\n📱 Testing Real Device (Samsung Galaxy S10+)...")
    real_device = create_real_device_sample()
    real_result = api.analyze_device(real_device)
    print_analysis_result("Real Device", real_result)
    
    # Test emulator
    print("\n🖥️ Testing Emulator (Android SDK x86_64)...")
    emulator_device = create_emulator_sample()
    emulator_result = api.analyze_device(emulator_device)
    print_analysis_result("Emulator", emulator_result)
    
    # Demonstrate model switching if multiple models available
    available_models = available.get('available_models', [])
    if len(available_models) > 1:
        print("\n🔄 Demonstrating Model Switching...")
        for model in ['random_forest', 'svm', 'gradient_boosting', 'logistic_regression']:
            if model in available_models:
                print(f"\nSwitching to {model}...")
                switch_result = api.switch_model(model)
                if switch_result['status'] == 'success':
                    print(f"✅ Successfully switched to {model}")
                    # Quick test with new model
                    test_result = api.analyze_device(emulator_device)
                    risk_level = test_result.get('result', {}).get('risk_level', 'unknown')
                    print(f"   Risk assessment: {risk_level}")
                else:
                    print(f"❌ Failed to switch: {switch_result.get('error', 'Unknown error')}")
                break

def demo_health_check():
    """Demonstrate health check"""
    print("\n🏥 HEALTH CHECK DEMO")
    
    api = ProductionAPI()
    health = api.health_check()
    
    print(f"Service Status: {'✅ HEALTHY' if health['status'] == 'healthy' else '❌ UNHEALTHY'}")
    print(f"Model Loaded: {'✅ YES' if health['model_loaded'] else '❌ NO'}")
    print(f"Model Name: {health['model_name']}")
    print(f"Timestamp: {health['timestamp']}")

def save_sample_files():
    """Save sample device files for testing"""
    samples_dir = Path("sample_devices")
    samples_dir.mkdir(exist_ok=True)
    
    # Save real device sample
    real_device = create_real_device_sample()
    with open(samples_dir / "real_device_samsung.json", 'w') as f:
        json.dump(real_device, f, indent=2)
    
    # Save emulator sample
    emulator_device = create_emulator_sample()
    with open(samples_dir / "emulator_x86.json", 'w') as f:
        json.dump(emulator_device, f, indent=2)
    
    print(f"\n💾 Sample files saved to {samples_dir}/")
    print("  - real_device_samsung.json")
    print("  - emulator_x86.json")

def main():
    """Run the demo"""
    try:
        demo_single_predictions()
        demo_health_check()
        save_sample_files()
        
        print(f"\n{'='*60}")
        print("✅ DEMO COMPLETED SUCCESSFULLY")
        print("📚 Check the README.md for API documentation")
        print("🧪 Run test_api.py to test the FastAPI endpoints")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        print("This might be due to missing dependencies.")
        print("Install required packages: pip install pandas numpy scikit-learn joblib")

if __name__ == "__main__":
    main()
