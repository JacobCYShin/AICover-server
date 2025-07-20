"""
RunPod Serverless Handler for AICoverGen
This is the entry point for RunPod serverless inference.
"""

import os
import json
import base64
import tempfile
import logging
from typing import Dict, Any, Optional
import torch
import numpy as np

# Add src to path
import sys
sys.path.append('/app/src')

from main import song_cover_pipeline
from uvr_integrated.separator import Separator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model caching
separator = None
available_voice_models = []

def init():
    """
    Initialize the model and load required resources.
    This function is called once when the container starts.
    """
    global separator, available_voice_models
    
    logger.info("🚀 Initializing AICoverGen Serverless Handler...")
    
    try:
        # Initialize UVR separator
        separator = Separator(
            output_dir="/app/temp",
            log_level=logging.INFO,
            info_only=False
        )
        
        # Get available voice models
        rvc_models_dir = "/app/rvc_models"
        if os.path.exists(rvc_models_dir):
            available_voice_models = [
                d for d in os.listdir(rvc_models_dir) 
                if os.path.isdir(os.path.join(rvc_models_dir, d)) 
                and d not in ['hubert_base.pt', 'MODELS.txt', 'public_models.json', 'rmvpe.pt']
            ]
        
        logger.info(f"✅ Initialization complete. Available voice models: {available_voice_models}")
        
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        raise

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main handler function for RunPod serverless inference.
    
    Args:
        event: Dictionary containing the input data
        
    Returns:
        Dictionary containing the response data
    """
    try:
        logger.info("📥 Received inference request")
        
        # Parse input
        input_data = event.get("input", {})
        
        # Validate required fields
        if "audio_data" not in input_data:
            return {
                "error": "Missing required field: audio_data"
            }
        
        if "voice_model" not in input_data:
            return {
                "error": "Missing required field: voice_model"
            }
        
        # Extract parameters
        audio_data = input_data["audio_data"]
        voice_model = input_data["voice_model"]
        pitch_change = input_data.get("pitch_change", 0)
        use_uvr = input_data.get("use_uvr", True)
        output_format = input_data.get("output_format", "mp3")
        
        # Validate voice model
        if voice_model not in available_voice_models:
            return {
                "error": f"Voice model '{voice_model}' not found. Available models: {available_voice_models}"
            }
        
        # Decode audio data
        try:
            if isinstance(audio_data, str):
                # Base64 encoded string
                audio_bytes = base64.b64decode(audio_data)
            else:
                # Already bytes
                audio_bytes = audio_data
        except Exception as e:
            return {
                "error": f"Failed to decode audio data: {e}"
            }
        
        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(suffix=f".{output_format}", delete=False) as temp_file:
            temp_file.write(audio_bytes)
            temp_audio_path = temp_file.name
        
        try:
            logger.info(f"🎵 Processing audio with voice model: {voice_model}")
            
            # Run the cover generation pipeline
            cover_path = song_cover_pipeline(
                song_input=temp_audio_path,
                voice_model=voice_model,
                pitch_change=pitch_change,
                keep_files=False,
                is_webui=0,
                use_uvr=use_uvr,
                output_format=output_format
            )
            
            # Read the generated cover
            with open(cover_path, 'rb') as f:
                cover_bytes = f.read()
            
            # Encode to base64
            cover_base64 = base64.b64encode(cover_bytes).decode('utf-8')
            
            # Clean up temporary files
            os.unlink(temp_audio_path)
            if os.path.exists(cover_path):
                os.unlink(cover_path)
            
            logger.info("✅ Cover generation completed successfully")
            
            return {
                "success": True,
                "cover_audio": cover_base64,
                "output_format": output_format,
                "voice_model": voice_model,
                "pitch_change": pitch_change,
                "use_uvr": use_uvr
            }
            
        except Exception as e:
            logger.error(f"❌ Cover generation failed: {e}")
            return {
                "error": f"Cover generation failed: {str(e)}"
            }
        finally:
            # Clean up temporary files
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
    
    except Exception as e:
        logger.error(f"❌ Handler error: {e}")
        return {
            "error": f"Handler error: {str(e)}"
        }

def get_available_models() -> Dict[str, Any]:
    """
    Get available voice models.
    
    Returns:
        Dictionary containing available models
    """
    return {
        "available_models": available_voice_models,
        "total_count": len(available_voice_models)
    }

# Health check endpoint
def health_check() -> Dict[str, Any]:
    """
    Health check function.
    
    Returns:
        Dictionary containing health status
    """
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "available_models": len(available_voice_models)
    } 