#!/usr/bin/env python3
"""
Test script for UVR integration with AICoverGen
"""

import os
import sys

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from main import song_cover_pipeline

def test_uvr_integration():
    """Test UVR integration with a sample audio file"""
    
    # Test parameters
    song_input = "test_audio.mp3"  # Replace with actual test file
    voice_model = "test_model"     # Replace with actual model directory
    pitch_change = 0
    keep_files = True
    use_uvr = True  # Enable UVR separation
    
    # Check if test file exists
    if not os.path.exists(song_input):
        print(f"Test file {song_input} not found. Please provide a test audio file.")
        return
    
    # Check if voice model exists
    rvc_models_dir = os.path.join(os.path.dirname(__file__), 'rvc_models')
    model_dir = os.path.join(rvc_models_dir, voice_model)
    if not os.path.exists(model_dir):
        print(f"Voice model directory {model_dir} not found. Please provide a valid model.")
        return
    
    try:
        print("Starting UVR integration test...")
        print(f"Input: {song_input}")
        print(f"Voice model: {voice_model}")
        print(f"Using UVR: {use_uvr}")
        
        # Run the pipeline
        cover_path = song_cover_pipeline(
            song_input=song_input,
            voice_model=voice_model,
            pitch_change=pitch_change,
            keep_files=keep_files,
            use_uvr=use_uvr,
            is_webui=0  # Command line mode
        )
        
        print(f"✅ Test completed successfully!")
        print(f"Cover generated at: {cover_path}")
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_uvr_integration() 