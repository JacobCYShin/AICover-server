"""
FastAPI Server for AICoverGen
Local development and testing server
"""

import os
import base64
import tempfile
import logging
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import torch

# Add src to path
import sys
sys.path.append('/app/src')

from main import song_cover_pipeline
from uvr_integrated.separator import Separator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AICoverGen API",
    description="AI Cover Generation API with integrated UVR separation",
    version="1.0.0"
)

# Global variables
separator = None
available_voice_models = []

# Pydantic models
class CoverRequest(BaseModel):
    voice_model: str
    pitch_change: int = 0
    use_uvr: bool = True
    output_format: str = "mp3"

class CoverResponse(BaseModel):
    success: bool
    cover_audio: str
    output_format: str
    voice_model: str
    pitch_change: int
    use_uvr: bool

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    global separator, available_voice_models
    
    logger.info("🚀 Starting AICoverGen API Server...")
    
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
        
        logger.info(f"✅ Server started. Available voice models: {available_voice_models}")
        
    except Exception as e:
        logger.error(f"❌ Server startup failed: {e}")
        raise

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "AICoverGen API Server",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "available_models": len(available_voice_models)
    }

@app.get("/models")
async def get_models():
    """Get available voice models."""
    return {
        "available_models": available_voice_models,
        "total_count": len(available_voice_models)
    }

@app.post("/generate", response_model=CoverResponse)
async def generate_cover(
    audio_file: UploadFile = File(...),
    voice_model: str = Form(...),
    pitch_change: int = Form(0),
    use_uvr: bool = Form(True),
    output_format: str = Form("mp3")
):
    """
    Generate AI cover from uploaded audio file.
    
    Args:
        audio_file: Audio file to process
        voice_model: Name of the voice model to use
        pitch_change: Pitch change in semitones
        use_uvr: Whether to use UVR separation
        output_format: Output format (mp3 or wav)
    
    Returns:
        CoverResponse with base64 encoded audio
    """
    try:
        # Validate voice model
        if voice_model not in available_voice_models:
            raise HTTPException(
                status_code=400, 
                detail=f"Voice model '{voice_model}' not found. Available models: {available_voice_models}"
            )
        
        # Validate file type
        if not audio_file.content_type.startswith('audio/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an audio file"
            )
        
        # Save uploaded file
        with tempfile.NamedTemporaryFile(suffix=f".{output_format}", delete=False) as temp_file:
            content = await audio_file.read()
            temp_file.write(content)
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
            
            logger.info("✅ Cover generation completed successfully")
            
            return CoverResponse(
                success=True,
                cover_audio=cover_base64,
                output_format=output_format,
                voice_model=voice_model,
                pitch_change=pitch_change,
                use_uvr=use_uvr
            )
            
        except Exception as e:
            logger.error(f"❌ Cover generation failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Cover generation failed: {str(e)}"
            )
        finally:
            # Clean up temporary files
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
            if 'cover_path' in locals() and os.path.exists(cover_path):
                os.unlink(cover_path)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ API error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/generate_base64")
async def generate_cover_base64(request: Dict[str, Any]):
    """
    Generate AI cover from base64 encoded audio data.
    
    Args:
        request: Dictionary containing audio_data (base64) and parameters
    
    Returns:
        Dictionary with base64 encoded cover audio
    """
    try:
        # Validate required fields
        if "audio_data" not in request:
            raise HTTPException(
                status_code=400,
                detail="Missing required field: audio_data"
            )
        
        if "voice_model" not in request:
            raise HTTPException(
                status_code=400,
                detail="Missing required field: voice_model"
            )
        
        # Extract parameters
        audio_data = request["audio_data"]
        voice_model = request["voice_model"]
        pitch_change = request.get("pitch_change", 0)
        use_uvr = request.get("use_uvr", True)
        output_format = request.get("output_format", "mp3")
        
        # Validate voice model
        if voice_model not in available_voice_models:
            raise HTTPException(
                status_code=400,
                detail=f"Voice model '{voice_model}' not found. Available models: {available_voice_models}"
            )
        
        # Decode audio data
        try:
            if isinstance(audio_data, str):
                audio_bytes = base64.b64decode(audio_data)
            else:
                audio_bytes = audio_data
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to decode audio data: {e}"
            )
        
        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(suffix=f".{output_format}", delete=False) as temp_file:
            temp_file.write(audio_bytes)
            temp_audio_path = temp_file.name
        
        try:
            logger.info(f"🎵 Processing base64 audio with voice model: {voice_model}")
            
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
            raise HTTPException(
                status_code=500,
                detail=f"Cover generation failed: {str(e)}"
            )
        finally:
            # Clean up temporary files
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
            if 'cover_path' in locals() and os.path.exists(cover_path):
                os.unlink(cover_path)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ API error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    ) 