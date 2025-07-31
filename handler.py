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
_initialized = False

def init_models():
    """
    Initialize the model and load required resources.
    This function is called lazily on first request.
    """
    global separator, available_voice_models, _initialized
    
    if _initialized:
        return
    
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
        _initialized = True
        
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        raise

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main handler function for RunPod serverless requests.
    """
    # Lazy initialization on first request
    if not _initialized:
        init_models()
    
    input_data = job.get("input", {})
    
    # Extract parameters
    rvc_dirname = input_data.get("rvc_dirname")
    song_input = input_data.get("song_input")
    pitch_change_all = input_data.get("pitch_change_all", 0)
    output_format = input_data.get("output_format", "mp3")
    
    # Validate required parameters
    if not rvc_dirname or not song_input:
        return {
            "error": "Missing required parameters: 'rvc_dirname' and 'song_input'",
            "available_models": available_voice_models
        }
    
    # Validate voice model
    if rvc_dirname not in available_voice_models:
        return {
            "error": f"Voice model '{rvc_dirname}' not found",
            "available_models": available_voice_models
        }
    
    try:
        logger.info(f"🎵 Processing cover generation request...")
        logger.info(f"📁 Voice Model: {rvc_dirname}")
        logger.info(f"🎧 Input: {song_input}")
        logger.info(f"🎹 Pitch Change: {pitch_change_all}")
        
        # Call the main pipeline
        result = song_cover_pipeline(
            rvc_dirname=rvc_dirname,
            song_input=song_input,
            pitch_change_all=pitch_change_all,
            keep_orig=False,
            main_gain=0,
            backup_gain=0,
            inst_gain=0,
            index_rate=0.5,
            filter_radius=3,
            rms_mix_rate=0.25,
            f0_method="rmvpe",
            crepe_hop_length=128,
            protect=0.33,
            pitch_change_vocals=0,
            pitch_change_inst=0,
            is_webui=False,
            progress=None,
            output_format=output_format,
            use_uvr=True  # Use UVR separation
        )
        
        if result and len(result) > 0:
            output_file = result[0] if isinstance(result, list) else result
            
            # Read output file and encode as base64
            if os.path.exists(output_file):
                with open(output_file, 'rb') as f:
                    audio_data = base64.b64encode(f.read()).decode('utf-8')
                
                file_size = os.path.getsize(output_file)
                
                return {
                    "success": True,
                    "output_file": os.path.basename(output_file),
                    "audio_data": audio_data,
                    "file_size": file_size,
                    "format": output_format
                }
            else:
                return {"error": "Output file not found"}
        else:
            return {"error": "Cover generation failed"}
            
    except Exception as e:
        logger.error(f"❌ Error in handler: {str(e)}")
        return {"error": f"Processing failed: {str(e)}"}

# RunPod serverless start
import runpod
runpod.serverless.start({"handler": handler}) 