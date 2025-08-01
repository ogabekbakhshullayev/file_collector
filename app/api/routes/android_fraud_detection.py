"""
Android Fraud Detection API Route

This module provides FastAPI endpoints for Android device fraud detection.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
import json
import logging
from pydantic import BaseModel
from enum import Enum

from fraud_detection.predictor import ProductionAPI

# Configure logging
logger = logging.getLogger(__name__)

# Initialize the fraud detection API lazily to avoid startup issues
fraud_api = None

class ModelName(str, Enum):
    """Available ML models for fraud detection"""
    XGBOOST = "xgboost"
    RANDOM_FOREST = "random_forest"
    SVM = "svm"
    GRADIENT_BOOSTING = "gradient_boosting"
    LOGISTIC_REGRESSION = "logistic_regression"
    MOCK = "mock"

# Initialize the fraud detection API lazily to avoid startup issues
fraud_api = None

def get_fraud_api():
    """Get or create the fraud detection API instance"""
    global fraud_api
    if fraud_api is None:
        fraud_api = ProductionAPI(model_name="xgboost")
    return fraud_api

router = APIRouter()

class DeviceMetadata(BaseModel):
    """Pydantic model for device metadata input"""
    device_model: Optional[str] = None
    manufacturer: Optional[str] = None
    device_brand: Optional[str] = None
    android_version: Optional[str] = None
    api_level: Optional[int] = None
    screen_width: Optional[int] = None
    screen_height: Optional[int] = None
    screen_density: Optional[int] = None
    cpu_info: Optional[Dict[str, Any]] = None
    total_memory: Optional[int] = None
    available_memory: Optional[int] = None
    internal_storage: Optional[int] = None
    external_storage: Optional[int] = None
    system_features: Optional[List[str]] = None
    installed_apps: Optional[List[Dict[str, Any]]] = None
    build_info: Optional[Dict[str, Any]] = None
    cameras: Optional[List[Dict[str, Any]]] = None
    sensors: Optional[List[Dict[str, Any]]] = None

class FraudDetectionRequest(BaseModel):
    """Request model for fraud detection"""
    device_metadata: DeviceMetadata
    model_name: Optional[ModelName] = ModelName.XGBOOST  # Default model, can be overridden

class FraudDetectionResponse(BaseModel):
    """Response model for fraud detection"""
    status: str
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    timestamp: Optional[str] = None
    model_used: Optional[str] = None  # Include which model was used

@router.post("/analyze-device", response_model=FraudDetectionResponse)
async def analyze_device_fraud(request: FraudDetectionRequest):
    """
    Analyze Android device metadata for fraud detection.
    
    This endpoint accepts device metadata and returns fraud analysis results
    including whether the device is likely to be an emulator or real device.
    
    You can optionally specify which ML model to use for analysis by including
    the model_name parameter. Available models: xgboost, random_forest, svm, 
    gradient_boosting, logistic_regression, mock.
    
    Args:
        request: Device metadata and optional model selection for analysis
        
    Returns:
        Analysis result with fraud detection prediction and confidence scores
    """
    try:
        # Get API instance and switch model if needed
        api = get_fraud_api()
        
        # Switch to requested model if different from current
        if request.model_name and request.model_name.value != api.predictor.model_name:
            switch_result = api.switch_model(request.model_name.value)
            if switch_result['status'] != 'success':
                return JSONResponse(
                    status_code=400,
                    content={
                        'status': 'error',
                        'error_message': f"Failed to switch to model '{request.model_name.value}': {switch_result.get('error', 'Unknown error')}",
                        'available_models': switch_result.get('available_models', []),
                        'timestamp': switch_result.get('timestamp')
                    }
                )
        
        # Convert Pydantic model to dict
        device_data = request.device_metadata.dict(exclude_none=True)
        
        # Analyze the device
        result = api.analyze_device(device_data)
        
        # Add model information to response
        if result.get('status') == 'success':
            result['model_used'] = api.predictor.model_name
        
        return JSONResponse(
            status_code=200,
            content=result
        )
        
    except Exception as e:
        logger.error(f"Error in analyze_device_fraud: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.post("/analyze-device-json")
async def analyze_device_from_json(file: UploadFile = File(...), model_name: Optional[ModelName] = ModelName.XGBOOST):
    """
    Analyze Android device from uploaded JSON file.
    
    This endpoint accepts a JSON file containing device metadata
    and returns fraud analysis results. You can optionally specify
    which ML model to use for analysis.
    
    Args:
        file: JSON file containing device metadata
        model_name: Optional ML model to use (ModelName enum: XGBOOST, RANDOM_FOREST, SVM, GRADIENT_BOOSTING, LOGISTIC_REGRESSION, MOCK)
        
    Returns:
        Analysis result with fraud detection prediction and confidence scores
    """
    try:
        # Validate file type
        if not file.filename.endswith('.json'):
            raise HTTPException(
                status_code=400,
                detail="File must be a JSON file"
            )
        
        # Get API instance and switch model if needed
        api = get_fraud_api()
        
        # Switch to requested model if different from current
        if model_name and model_name.value != api.predictor.model_name:
            switch_result = api.switch_model(model_name.value)
            if switch_result['status'] != 'success':
                return JSONResponse(
                    status_code=400,
                    content={
                        'status': 'error',
                        'error_message': f"Failed to switch to model '{model_name.value}': {switch_result.get('error', 'Unknown error')}",
                        'available_models': switch_result.get('available_models', []),
                        'timestamp': switch_result.get('timestamp')
                    }
                )
        
        # Read and parse JSON file
        content = await file.read()
        try:
            device_data = json.loads(content.decode('utf-8'))
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid JSON file: {str(e)}"
            )
        
        # Analyze the device
        result = api.analyze_device(device_data)
        
        # Add model and file information to response
        if result.get('status') == 'success':
            result['model_used'] = api.predictor.model_name
            result['filename'] = file.filename
        
        return JSONResponse(
            status_code=200,
            content=result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in analyze_device_from_json: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.post("/batch-analyze")
async def batch_analyze_devices(files: List[UploadFile] = File(...), model_name: Optional[ModelName] = ModelName.XGBOOST):
    """
    Analyze multiple Android devices from uploaded JSON files.
    
    This endpoint accepts multiple JSON files containing device metadata
    and returns fraud analysis results for each device. You can optionally
    specify which ML model to use for all analyses.
    
    Args:
        files: List of JSON files containing device metadata
        model_name: Optional ML model to use (ModelName enum: XGBOOST, RANDOM_FOREST, SVM, GRADIENT_BOOSTING, LOGISTIC_REGRESSION, MOCK)
        
    Returns:
        List of analysis results for each device
    """
    try:
        # Get API instance and switch model if needed
        api = get_fraud_api()
        
        # Switch to requested model if different from current
        if model_name and model_name.value != api.predictor.model_name:
            switch_result = api.switch_model(model_name.value)
            if switch_result['status'] != 'success':
                return JSONResponse(
                    status_code=400,
                    content={
                        'status': 'error',
                        'error_message': f"Failed to switch to model '{model_name.value}': {switch_result.get('error', 'Unknown error')}",
                        'available_models': switch_result.get('available_models', []),
                        'timestamp': switch_result.get('timestamp')
                    }
                )
        
        results = []
        
        for file in files:
            try:
                # Validate file type
                if not file.filename.endswith('.json'):
                    results.append({
                        'filename': file.filename,
                        'status': 'error',
                        'error_message': 'File must be a JSON file'
                    })
                    continue
                
                # Read and parse JSON file
                content = await file.read()
                try:
                    device_data = json.loads(content.decode('utf-8'))
                except json.JSONDecodeError as e:
                    results.append({
                        'filename': file.filename,
                        'status': 'error',
                        'error_message': f'Invalid JSON file: {str(e)}'
                    })
                    continue
                
                # Analyze the device
                result = api.analyze_device(device_data)
                result['filename'] = file.filename
                result['model_used'] = api.predictor.model_name
                results.append(result)
                
            except Exception as e:
                results.append({
                    'filename': file.filename,
                    'status': 'error',
                    'error_message': str(e)
                })
        
        return JSONResponse(
            status_code=200,
            content={
                'status': 'success',
                'total_files': len(files),
                'model_used': api.predictor.model_name,
                'results': results
            }
        )
        
    except Exception as e:
        logger.error(f"Error in batch_analyze_devices: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

