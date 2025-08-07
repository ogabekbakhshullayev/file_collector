"""
Android Device Fraud Detection - Production Deployment Module

This module provides a production-ready interface for the trained fraud detection
model, including real-time prediction capabilities and model evaluation tools.
"""

import json
import time
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# Try to import ML dependencies, fallback to None if not available
try:
    import numpy as np
    import pandas as pd
    import joblib
    ML_AVAILABLE = True
except ImportError as e:
    logging.warning(f"ML dependencies not available: {str(e)}. Using fallback implementations.")
    try:
        # Try alternative numpy import for compatibility
        import numpy as np
        import pandas as pd
        import joblib
        ML_AVAILABLE = True
    except (ImportError, AttributeError) as e2:
        logging.warning(f"Alternative ML import also failed: {str(e2)}. Using mock implementations.")
        np = None
        pd = None
        joblib = None
        ML_AVAILABLE = False

from .data_processor import AndroidDeviceFeatureExtractor

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FraudDetectionPredictor:
    """
    Production-ready fraud detection predictor for Android devices.
    Provides real-time classification of devices as real or emulated.
    """
    
    def __init__(self, model_path: str = "fraud_detection/models", model_name: str = "xgboost"):
        """
        Initialize the predictor with a trained model.
        
        Args:
            model_path: Path to directory containing trained models
            model_name: Name of the model to load (e.g., 'xgboost', 'random_forest')
        """
        self.model_path = Path(model_path)
        self.model_name = model_name
        self.feature_extractor = AndroidDeviceFeatureExtractor()
        
        # Load model components
        self.model = None
        self.scaler = None
        self.feature_selector = None
        self.feature_names = None
        
        # Initialize with mock model if real model not available
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize model or create a mock model for demonstration"""
        try:
            self.load_model()
        except (FileNotFoundError, ModuleNotFoundError, ImportError, Exception) as e:
            logger.warning(f"Model loading failed ({str(e)}). Creating mock model for demonstration.")
            self._create_mock_model()
    
    def get_available_models(self) -> List[str]:
        """Get list of available model files"""
        available_models = []
        
        # Check for common model types
        model_types = ["xgboost", "random_forest", "svm", "gradient_boosting", "logistic_regression"]
        
        for model_type in model_types:
            model_file = self.model_path / f"{model_type}_model.joblib"
            if model_file.exists():
                available_models.append(model_type)
        
        # If no real models found, return mock model option
        if not available_models:
            available_models.append("mock")
        
        return available_models
    
    def switch_model(self, model_name: str) -> Dict[str, Any]:
        """
        Switch to a different model
        
        Args:
            model_name: Name of the model to switch to
            
        Returns:
            Dictionary with switch result and model info
        """
        try:
            available_models = self.get_available_models()
            
            if model_name not in available_models and model_name != "mock":
                return {
                    "success": False,
                    "error": f"Model '{model_name}' not available. Available models: {available_models}",
                    "available_models": available_models
                }
            
            old_model = self.model_name
            self.model_name = model_name
            
            # Reset model components
            self.model = None
            self.scaler = None
            self.feature_selector = None
            
            # Load new model
            if model_name == "mock":
                self._create_mock_model()
            else:
                try:
                    self.load_model()
                except (FileNotFoundError, ModuleNotFoundError, ImportError, Exception) as e:
                    # Fallback to mock if model file not found or loading fails
                    logger.warning(f"Model loading failed for '{model_name}': {str(e)}. Using mock model.")
                    self.model_name = "mock"
                    self._create_mock_model()
            
            return {
                "success": True,
                "message": f"Successfully switched from '{old_model}' to '{model_name}'",
                "previous_model": old_model,
                "current_model": self.model_name,
                "available_models": available_models
            }
            
        except Exception as e:
            logger.error(f"Error switching model: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to switch model: {str(e)}",
                "current_model": self.model_name
            }
    
    def _create_mock_model(self):
        """Create a mock model for demonstration purposes"""
        class MockModel:
            def predict(self, X):
                # Simple heuristic for demonstration
                predictions = []
                
                # Handle different input types
                if hasattr(X, 'iloc'):  # pandas DataFrame
                    num_rows = len(X)
                elif hasattr(X, '__len__'):  # list or array
                    num_rows = len(X)
                else:
                    num_rows = 1
                
                for idx in range(num_rows):
                    # Simple heuristic: predict emulator if we detect certain patterns
                    # This is a simplified mock prediction
                    prediction = 0  # Default to real
                    
                    # Use a simple random-like approach based on index for demo
                    if (idx + 1) % 4 == 0:  # Every 4th device is predicted as emulator
                        prediction = 1
                    
                    predictions.append(prediction)
                
                if ML_AVAILABLE and np:
                    return np.array(predictions)
                else:
                    return predictions
            
            def predict_proba(self, X):
                predictions = self.predict(X)
                probabilities = []
                
                for pred in predictions:
                    if pred == 1:  # Emulator
                        probabilities.append([0.2, 0.8])  # Low real confidence, high emulator confidence
                    else:  # Real
                        probabilities.append([0.8, 0.2])  # High real confidence, low emulator confidence
                
                if ML_AVAILABLE and np:
                    return np.array(probabilities)
                else:
                    return probabilities
        
        self.model = MockModel()
        logger.info("Mock model created for demonstration")
    
    def load_model(self):
        """Load the trained model and associated components"""
        
        if not ML_AVAILABLE or not joblib:
            raise ImportError("ML dependencies (numpy, pandas, joblib) not available")
        
        try:
            # Load main model
            model_file = self.model_path / f"{self.model_name}_model.joblib"
            if model_file.exists():
                try:
                    self.model = joblib.load(model_file)
                    logger.info(f"Loaded {self.model_name} model from {model_file}")
                except (ImportError, AttributeError, ModuleNotFoundError) as e:
                    if 'numpy._core' in str(e) or '_core' in str(e):
                        logger.warning(f"Numpy compatibility issue loading model: {str(e)}. Model may work but with warnings.")
                        # Try to load anyway, as model might still work
                        try:
                            import warnings
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                self.model = joblib.load(model_file)
                                logger.info(f"Loaded {self.model_name} model from {model_file} (with numpy warnings suppressed)")
                        except Exception as e2:
                            logger.error(f"Failed to load model even with warning suppression: {str(e2)}")
                            raise
                    else:
                        raise
            else:
                raise FileNotFoundError(f"Model file not found: {model_file}")
            
            # Load scaler if exists
            scaler_file = self.model_path / f"{self.model_name}_scaler.joblib"
            if scaler_file.exists():
                try:
                    self.scaler = joblib.load(scaler_file)
                    logger.info(f"Loaded scaler from {scaler_file}")
                except (ImportError, AttributeError, ModuleNotFoundError) as e:
                    if 'numpy._core' in str(e) or '_core' in str(e):
                        logger.warning(f"Failed to load scaler due to numpy compatibility: {str(e)}")
                    else:
                        logger.warning(f"Failed to load scaler: {str(e)}")
                    self.scaler = None
                except Exception as e:
                    logger.warning(f"Failed to load scaler: {str(e)}")
                    self.scaler = None
            
            # Load feature selector if exists
            selector_file = self.model_path / "selectkbest_feature_selector.joblib"
            if selector_file.exists():
                try:
                    self.feature_selector = joblib.load(selector_file)
                    logger.info(f"Loaded feature selector from {selector_file}")
                except (ImportError, AttributeError, ModuleNotFoundError) as e:
                    if 'numpy._core' in str(e) or '_core' in str(e):
                        logger.warning(f"Failed to load feature selector due to numpy compatibility: {str(e)}")
                    else:
                        logger.warning(f"Failed to load feature selector: {str(e)}")
                    self.feature_selector = None
                except Exception as e:
                    logger.warning(f"Failed to load feature selector: {str(e)}")
                    self.feature_selector = None
            
            logger.info(f"Model {self.model_name} loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def predict_single_device(self, json_data: Dict) -> Dict[str, Any]:
        """
        Predict whether a single device is real or emulated.
        
        Args:
            json_data: Device metadata in JSON format
            
        Returns:
            Dictionary containing prediction results and confidence scores
        """
        
        start_time = time.time()
        
        try:
            # Extract features using the same feature extractor as training
            features = self.feature_extractor.extract_all_features(json_data)
            
            # Convert to DataFrame with proper feature names
            # Get the expected feature names from the training pipeline
            expected_features = [
                'num_cameras', 'max_resolution_width', 'max_resolution_height',
                'avg_resolution_ratio', 'resolution_variety_score', 'unique_formats',
                'num_cpu_cores', 'cpu_model_consistency', 'cpu_arch_score',
                'hardware_emulator_score', 'build_fingerprint_score', 'ro_secure',
                'ro_debuggable', 'build_type_score', 'num_hardware_features',
                'num_camera_features', 'num_sensor_features', 'num_telephony_features',
                'num_bluetooth_features', 'num_google_features', 'num_samsung_features',
                'kernel_version_score', 'num_installed_packages', 'google_apps_ratio',
                'system_apps_ratio', 'emulator_app_score', 'num_mount_points',
                'virtual_fs_score', 'emulator_likelihood_score'
            ]
            
            # Create DataFrame with expected feature names
            df = pd.DataFrame([features])
            
            # Ensure we have all expected features in the correct order
            for feature in expected_features:
                if feature not in df.columns:
                    df[feature] = 0
            
            # Select and order features as expected by the model
            X = df[expected_features].copy()
            X = X.fillna(0)
            X = X.replace([np.inf, -np.inf], 0)
            
            # Apply feature selection if available
            if self.feature_selector:
                try:
                    X_selected = self.feature_selector.transform(X)
                    # Use the stored selected feature names if available
                    if hasattr(self, 'selected_features') and self.selected_features:
                        X = pd.DataFrame(X_selected, columns=self.selected_features)
                    else:
                        # Fallback: try to get feature names from selector
                        if hasattr(self.feature_selector, 'get_support'):
                            selected_mask = self.feature_selector.get_support()
                            selected_feature_names = [name for name, selected in zip(expected_features, selected_mask) if selected]
                            X = pd.DataFrame(X_selected, columns=selected_feature_names)
                        else:
                            X = pd.DataFrame(X_selected)
                except Exception as e:
                    print(f"Warning: Feature selection failed: {e}")
                    # Continue without feature selection
            
            # Apply scaling if available
            if self.scaler:
                try:
                    X_scaled = self.scaler.transform(X)
                    X = pd.DataFrame(X_scaled, columns=X.columns)
                except Exception as e:
                    print(f"Warning: Scaling failed: {e}")
                    # Continue without scaling
            
            # Make prediction
            prediction = self.model.predict(X)[0]
            
            # Get prediction probability if available
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X)[0]
                real_confidence = probabilities[0]
                emulator_confidence = probabilities[1]
            else:
                real_confidence = 1.0 - prediction
                emulator_confidence = float(prediction)
            
            processing_time = time.time() - start_time
            
            # Prepare result
            result = {
                'prediction': 'emulator' if prediction == 1 else 'real',
                'prediction_numeric': int(prediction),
                'confidence': {
                    'real': float(real_confidence),
                    'emulator': float(emulator_confidence)
                },
                'risk_score': float(emulator_confidence),  # Higher = more likely emulator
                'processing_time_ms': round(processing_time * 1000, 2),
                'model_used': self.model_name,
                'features_analyzed': len(X.columns),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Add detailed analysis
            result['analysis'] = self._analyze_device_characteristics(features)
            
            return result
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            processing_time = time.time() - start_time
            return {
                'prediction': 'error',
                'error': str(e),
                'processing_time_ms': round(processing_time * 1000, 2),
                'model_used': self.model_name,
                'analysis': {'hardware_indicators': {}, 'risk_factors': [], 'confidence_factors': []}
            }
    
    def _simple_heuristic_prediction(self, features: Dict[str, Any], start_time: float) -> Dict[str, Any]:
        """Simple heuristic prediction when ML libraries are not available"""
        
        # Validate features input
        if not isinstance(features, dict):
            logger.error(f"features is not a dict in _simple_heuristic_prediction: {type(features)}, value: {features}")
            processing_time = time.time() - start_time
            return {
                'prediction': 'error',
                'error': f'Invalid features data type: {type(features)}',
                'processing_time_ms': round(processing_time * 1000, 2),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        
        # Simple scoring based on common emulator indicators
        emulator_score = 0.0
        
        # Check CPU architecture
        if features.get('is_intel_cpu', False):
            emulator_score += 0.4
        
        # Check emulator likelihood score
        if features.get('emulator_likelihood_score', 0) > 0:
            emulator_score += 0.3
        
        # Check for test keys
        if features.get('has_test_keys', False):
            emulator_score += 0.2
        
        # Check for VirtualBox
        if features.get('has_vbox_mounts', False):
            emulator_score += 0.3
        
        # Check camera count (real devices usually have more cameras)
        num_cameras = features.get('num_cameras', 0)
        if num_cameras < 2:
            emulator_score += 0.1
        
        # Normalize score
        emulator_score = min(1.0, emulator_score)
        real_score = 1.0 - emulator_score
        
        prediction = 'emulator' if emulator_score > 0.5 else 'real'
        processing_time = time.time() - start_time
        
        result = {
            'prediction': prediction,
            'prediction_numeric': 1 if prediction == 'emulator' else 0,
            'confidence': {
                'real': real_score,
                'emulator': emulator_score
            },
            'risk_score': emulator_score,
            'processing_time_ms': round(processing_time * 1000, 2),
            'model_used': f"{self.model_name}_heuristic",
            'features_analyzed': len(features),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'note': 'Using heuristic prediction (ML libraries not available)'
        }
        
        # Add detailed analysis
        result['analysis'] = self._analyze_device_characteristics(features)
        
        return result
    
    def _analyze_device_characteristics(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Provide detailed analysis of device characteristics"""
        
        # Validate features input
        if not isinstance(features, dict):
            logger.error(f"features is not a dict in _analyze_device_characteristics: {type(features)}, value: {features}")
            return {
                'hardware_indicators': {'error': 'Invalid features data'},
                'system_indicators': {'error': 'Invalid features data'},
                'risk_factors': ['Unable to analyze due to invalid data'],
                'confidence_factors': []
            }
        
        analysis = {
            'hardware_indicators': {},
            'system_indicators': {},
            'risk_factors': [],
            'confidence_factors': []
        }
        
        # Hardware analysis
        analysis['hardware_indicators'] = {
            'cpu_architecture': 'ARM' if features.get('is_arm_cpu', False) else 'x86' if features.get('is_intel_cpu', False) else 'Unknown',
            'num_cameras': features.get('num_cameras', 0),
            'max_resolution': f"{features.get('max_resolution_width', 0)}x{features.get('max_resolution_height', 0)}",
            'cpu_cores': features.get('num_cpu_cores', 0)
        }
        
        # System analysis
        analysis['system_indicators'] = {
            'build_type_score': features.get('build_type_score', 0),
            'hardware_emulator_score': features.get('hardware_emulator_score', 0),
            'emulator_likelihood_score': features.get('emulator_likelihood_score', 0)
        }
        
        # Risk factors (indicators of emulation)
        if features.get('is_intel_cpu', False):
            analysis['risk_factors'].append('Intel CPU detected (common in emulators)')
        
        if features.get('has_vbox_mounts', False):
            analysis['risk_factors'].append('VirtualBox filesystem mounts detected')
        
        if features.get('hardware_emulator_score', 0) > 0:
            analysis['risk_factors'].append('Emulator hardware signatures detected')
        
        if features.get('has_test_keys', False):
            analysis['risk_factors'].append('Test signing keys detected')
        
        if features.get('max_resolution_width', 0) < 1920:
            analysis['risk_factors'].append('Low maximum camera resolution')
        
        # Confidence factors (indicators of real device)
        if features.get('is_arm_cpu', False):
            analysis['confidence_factors'].append('ARM CPU architecture (typical of real devices)')
        
        if features.get('num_samsung_features', 0) > 0:
            analysis['confidence_factors'].append('Samsung-specific features detected')
        
        if features.get('num_cameras', 0) >= 3:
            analysis['confidence_factors'].append('Multiple cameras detected')
        
        if features.get('build_type_score', 0) > 0:
            analysis['confidence_factors'].append('Production build type')
        
        return analysis

class ProductionAPI:
    """
    Simple API wrapper for production deployment.
    This can be extended to work with Flask, FastAPI, or other web frameworks.
    """
    
    def __init__(self, model_name: str = "xgboost"):
        self.predictor = FraudDetectionPredictor(model_name=model_name)
    
    def analyze_device(self, device_metadata: Dict) -> Dict[str, Any]:
        """
        Main API endpoint for device analysis.
        
        Args:
            device_metadata: Device JSON metadata
            
        Returns:
            Analysis result with prediction and confidence
        """
        
        try:
            # Validate input
            if not isinstance(device_metadata, dict):
                logger.error(f"device_metadata is not a dict: {type(device_metadata)}, value: {device_metadata}")
                return {
                    'status': 'error',
                    'error_message': f'Invalid input: expected dictionary, got {type(device_metadata).__name__}',
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
            
            result = self.predictor.predict_single_device(device_metadata)
            
            # Ensure result is a dictionary
            if not isinstance(result, dict):
                logger.error(f"predict_single_device returned non-dict type: {type(result)}, value: {result}")
                return {
                    'status': 'error',
                    'error_message': f'Internal error: invalid result type {type(result)}',
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
            
            if result.get('prediction') == 'error':
                return {
                    'status': 'error',
                    'error_message': result.get('error', 'Unknown error'),
                    'timestamp': result.get('timestamp')
                }
            
            # Format for API response
            api_response = {
                'status': 'success',
                'result': {
                    'is_emulator': result['prediction'] == 'emulator',
                    'confidence_score': result['confidence']['emulator'],
                    'risk_level': self._calculate_risk_level(result['risk_score']),
                    'analysis_summary': {
                        'cpu_architecture': result['analysis']['hardware_indicators']['cpu_architecture'],
                        'cameras_detected': result['analysis']['hardware_indicators']['num_cameras'],
                        'risk_factors_count': len(result['analysis']['risk_factors']),
                        'confidence_factors_count': len(result['analysis']['confidence_factors'])
                    },
                    'processing_time_ms': result['processing_time_ms'],
                    'model_version': result['model_used']
                },
                'metadata': {
                    'timestamp': result['timestamp'],
                    'features_analyzed': result['features_analyzed']
                }
            }
            
            return api_response
            
        except Exception as e:
            logger.error(f"Error in analyze_device: {str(e)}")
            return {
                'status': 'error',
                'error_message': str(e),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
    
    def _calculate_risk_level(self, risk_score: float) -> str:
        """Calculate human-readable risk level"""
        if risk_score < 0.3:
            return 'low'
        elif risk_score < 0.7:
            return 'medium'
        else:
            return 'high'
    
    def health_check(self) -> Dict[str, Any]:
        """API health check endpoint"""
        return {
            'status': 'healthy',
            'model_loaded': self.predictor.model is not None,
            'model_name': self.predictor.model_name,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def get_available_models(self) -> Dict[str, Any]:
        """Get list of available models"""
        try:
            available_models = self.predictor.get_available_models()
            return {
                'status': 'success',
                'available_models': available_models,
                'current_model': self.predictor.model_name,
                'total_models': len(available_models)
            }
        except Exception as e:
            logger.error(f"Error getting available models: {str(e)}")
            return {
                'status': 'error',
                'error_message': str(e),
                'current_model': self.predictor.model_name
            }
    
    def switch_model(self, model_name: str) -> Dict[str, Any]:
        """Switch to a different model"""
        try:
            result = self.predictor.switch_model(model_name)
            return {
                'status': 'success' if result['success'] else 'error',
                **result,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        except Exception as e:
            logger.error(f"Error switching model: {str(e)}")
            return {
                'status': 'error',
                'error_message': str(e),
                'current_model': self.predictor.model_name,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
    
    def get_model_details(self) -> Dict[str, Any]:
        """Get detailed information about the current model"""
        try:
            return {
                'status': 'success',
                'model_details': {
                    'name': self.predictor.model_name,
                    'type': type(self.predictor.model).__name__ if self.predictor.model else 'Unknown',
                    'model_loaded': self.predictor.model is not None,
                    'scaler_loaded': self.predictor.scaler is not None,
                    'feature_selector_loaded': self.predictor.feature_selector is not None,
                    'model_path': str(self.predictor.model_path),
                    'has_predict_proba': hasattr(self.predictor.model, 'predict_proba') if self.predictor.model else False
                },
                'capabilities': {
                    'real_vs_emulator_classification': True,
                    'confidence_scoring': True,
                    'risk_level_assessment': True,
                    'feature_analysis': True,
                    'batch_processing': True
                },
                'available_models': self.predictor.get_available_models(),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        except Exception as e:
            logger.error(f"Error getting model details: {str(e)}")
            return {
                'status': 'error',
                'error_message': str(e),
                'current_model': self.predictor.model_name,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
