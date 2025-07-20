#!/usr/bin/env python3
"""
AICoverGen API Client Example
Demonstrates how to use the AICoverGen API for cover generation
"""

import requests
import base64
import json
import os
from typing import Dict, Any

class AICoverGenClient:
    """Client for AICoverGen API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health"""
        response = requests.get(f"{self.base_url}/health")
        return response.json()
    
    def get_models(self) -> Dict[str, Any]:
        """Get available voice models"""
        response = requests.get(f"{self.base_url}/models")
        return response.json()
    
    def generate_cover_file(self, 
                          audio_file_path: str, 
                          voice_model: str, 
                          pitch_change: int = 0,
                          use_uvr: bool = True,
                          output_format: str = "mp3") -> Dict[str, Any]:
        """
        Generate cover from audio file
        
        Args:
            audio_file_path: Path to input audio file
            voice_model: Name of voice model to use
            pitch_change: Pitch change in semitones
            use_uvr: Whether to use UVR separation
            output_format: Output format (mp3 or wav)
        
        Returns:
            API response with base64 encoded cover audio
        """
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
        
        with open(audio_file_path, 'rb') as f:
            files = {'audio_file': f}
            data = {
                'voice_model': voice_model,
                'pitch_change': pitch_change,
                'use_uvr': use_uvr,
                'output_format': output_format
            }
            
            response = requests.post(f"{self.base_url}/generate", files=files, data=data)
            return response.json()
    
    def generate_cover_base64(self, 
                            audio_base64: str, 
                            voice_model: str, 
                            pitch_change: int = 0,
                            use_uvr: bool = True,
                            output_format: str = "mp3") -> Dict[str, Any]:
        """
        Generate cover from base64 encoded audio
        
        Args:
            audio_base64: Base64 encoded audio data
            voice_model: Name of voice model to use
            pitch_change: Pitch change in semitones
            use_uvr: Whether to use UVR separation
            output_format: Output format (mp3 or wav)
        
        Returns:
            API response with base64 encoded cover audio
        """
        data = {
            'audio_data': audio_base64,
            'voice_model': voice_model,
            'pitch_change': pitch_change,
            'use_uvr': use_uvr,
            'output_format': output_format
        }
        
        response = requests.post(f"{self.base_url}/generate_base64", json=data)
        return response.json()
    
    def save_cover_audio(self, response: Dict[str, Any], output_path: str):
        """
        Save cover audio from API response
        
        Args:
            response: API response containing base64 audio
            output_path: Path to save the audio file
        """
        if response.get('success') and 'cover_audio' in response:
            audio_data = base64.b64decode(response['cover_audio'])
            with open(output_path, 'wb') as f:
                f.write(audio_data)
            print(f"✅ Cover saved to: {output_path}")
        else:
            print(f"❌ Failed to save cover: {response.get('error', 'Unknown error')}")

def main():
    """Example usage of AICoverGen API client"""
    
    # Initialize client
    client = AICoverGenClient()
    
    print("🎵 AICoverGen API Client Example")
    print("=" * 50)
    
    # Check API health
    print("\n1. Checking API health...")
    try:
        health = client.health_check()
        print(f"✅ API Status: {health['status']}")
        print(f"   GPU Available: {health['gpu_available']}")
        print(f"   Available Models: {health['available_models']}")
    except Exception as e:
        print(f"❌ API health check failed: {e}")
        return
    
    # Get available models
    print("\n2. Getting available voice models...")
    try:
        models = client.get_models()
        print(f"✅ Available models: {models['available_models']}")
        print(f"   Total count: {models['total_count']}")
    except Exception as e:
        print(f"❌ Failed to get models: {e}")
        return
    
    # Example: Generate cover from file
    print("\n3. Example: Generate cover from audio file")
    print("   (This requires an actual audio file)")
    
    # Example file path - replace with actual file
    example_audio_file = "example_song.mp3"
    
    if os.path.exists(example_audio_file):
        try:
            # Use first available model
            if models['available_models']:
                voice_model = models['available_models'][0]
                
                print(f"   Using voice model: {voice_model}")
                print(f"   Processing file: {example_audio_file}")
                
                response = client.generate_cover_file(
                    audio_file_path=example_audio_file,
                    voice_model=voice_model,
                    pitch_change=0,
                    use_uvr=True,
                    output_format="mp3"
                )
                
                if response.get('success'):
                    output_path = f"generated_cover_{voice_model}.mp3"
                    client.save_cover_audio(response, output_path)
                else:
                    print(f"❌ Generation failed: {response.get('error')}")
            else:
                print("❌ No voice models available")
        except Exception as e:
            print(f"❌ Generation failed: {e}")
    else:
        print(f"   ⚠️  Example file not found: {example_audio_file}")
        print("   Create an audio file or update the path to test")
    
    # Example: Generate cover from base64
    print("\n4. Example: Generate cover from base64 audio")
    print("   (This requires base64 encoded audio data)")
    
    # Example base64 audio - replace with actual data
    example_base64_audio = "base64_encoded_audio_data_here"
    
    if example_base64_audio != "base64_encoded_audio_data_here" and models['available_models']:
        try:
            voice_model = models['available_models'][0]
            
            response = client.generate_cover_base64(
                audio_base64=example_base64_audio,
                voice_model=voice_model,
                pitch_change=0,
                use_uvr=True,
                output_format="mp3"
            )
            
            if response.get('success'):
                output_path = f"generated_cover_base64_{voice_model}.mp3"
                client.save_cover_audio(response, output_path)
            else:
                print(f"❌ Generation failed: {response.get('error')}")
        except Exception as e:
            print(f"❌ Generation failed: {e}")
    else:
        print("   ⚠️  No base64 audio data provided")
        print("   Update the example_base64_audio variable to test")
    
    print("\n" + "=" * 50)
    print("🎉 Client example completed!")
    print("\n💡 Usage tips:")
    print("- Ensure the API server is running")
    print("- Add voice models to the rvc_models directory")
    print("- Use supported audio formats (mp3, wav, etc.)")
    print("- Check the API documentation at http://localhost:8000/docs")

if __name__ == "__main__":
    main() 