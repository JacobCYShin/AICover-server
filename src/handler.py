"""RunPod Serverless Handler for AICoverGen (Official Template Style)"""

import runpod
import os
import base64
import tempfile
import logging
from typing import Dict, Any
import torch
import sys
sys.path.append('/app/src')
from main import song_cover_pipeline
from uvr_integrated.separator import Separator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model cache (if needed)
separator = None
available_voice_models = []

try:
    separator = Separator(
        output_dir="/app/temp",
        log_level=logging.INFO,
        info_only=False
    )
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

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    input_data = job.get("input", {})
    if "audio_data" not in input_data:
        return {"error": "Missing required field: audio_data"}
    if "voice_model" not in input_data:
        return {"error": "Missing required field: voice_model"}
    audio_data = input_data["audio_data"]
    voice_model = input_data["voice_model"]
    pitch_change = input_data.get("pitch_change", 0)
    use_uvr = input_data.get("use_uvr", True)
    output_format = input_data.get("output_format", "mp3")
    if voice_model not in available_voice_models:
        return {"error": f"Voice model '{voice_model}' not found. Available models: {available_voice_models}"}
    try:
        if isinstance(audio_data, str):
            audio_bytes = base64.b64decode(audio_data)
        else:
            audio_bytes = audio_data
    except Exception as e:
        return {"error": f"Failed to decode audio data: {e}"}
    with tempfile.NamedTemporaryFile(suffix=f".{output_format}", delete=False) as temp_file:
        temp_file.write(audio_bytes)
        temp_audio_path = temp_file.name
    try:
        logger.info(f"🎵 Processing audio with voice model: {voice_model}")
        cover_path = song_cover_pipeline(
            song_input=temp_audio_path,
            voice_model=voice_model,
            pitch_change=pitch_change,
            keep_files=False,
            is_webui=0,
            use_uvr=use_uvr,
            output_format=output_format
        )
        with open(cover_path, 'rb') as f:
            cover_bytes = f.read()
        cover_base64 = base64.b64encode(cover_bytes).decode('utf-8')
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
        return {"error": f"Cover generation failed: {str(e)}"}
    finally:
        if os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)

runpod.serverless.start({"handler": handler}) 