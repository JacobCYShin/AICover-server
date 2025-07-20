import os
import sys
import gc
import hashlib
import librosa
import numpy as np
import soundfile as sf
from urllib.parse import urlparse, parse_qs

# Import integrated UVR separator
from uvr_integrated.separator import Separator


def get_youtube_video_id(url, ignore_playlist=True):
    """
    Extract YouTube video ID from various YouTube URL formats
    """
    query = urlparse(url)
    if query.hostname == 'youtu.be':
        if query.path[1:] == 'watch':
            return query.query[2:]
        return query.path[1:]

    if query.hostname in {'www.youtube.com', 'youtube.com', 'music.youtube.com'}:
        if not ignore_playlist:
            # use case: get playlist id not current video in playlist
            try:
                return parse_qs(query.query)['list'][0]
            except KeyError:
                pass
        if query.path == '/watch':
            return parse_qs(query.query)['v'][0]
        if query.path[:7] == '/watch/':
            return query.path.split('/')[1]
        if query.path[:7] == '/embed/':
            return query.path.split('/')[2]
        if query.path[:3] == '/v/':
            return query.path.split('/')[2]

    # returns None for invalid YouTube url
    return None


def get_hash(filepath):
    """Generate hash for file identification"""
    with open(filepath, 'rb') as f:
        file_hash = hashlib.blake2b()
        while chunk := f.read(8192):
            file_hash.update(chunk)
    return file_hash.hexdigest()[:11]


def display_progress(message, percent, is_webui, progress=None):
    """Display progress message"""
    if is_webui:
        progress(percent, desc=message)
    else:
        print(message)


def run_uvr_separation(song_input, output_dir, song_id, is_webui, input_type, progress=None):
    """
    Run UVR separation pipeline using python-audio-separator
    
    This function follows the separation flow from test.py:
    1. Vocals and Instrumental separation
    2. Lead Vocals and Backing Vocals separation  
    3. DeReverb processing on Lead Vocals
    4. Denoise processing on Lead Vocals
    """
    
    # Initialize separator
    separator = Separator(output_dir=output_dir)
    
    # Define output file paths
    vocals_path = os.path.join(output_dir, 'Vocals.wav')
    instrumental_path = os.path.join(output_dir, 'Instrumental.wav')
    lead_vocals_path = os.path.join(output_dir, 'Lead_Vocals.wav')
    backing_vocals_path = os.path.join(output_dir, 'Backing_Vocals.wav')
    lead_vocals_reverb_path = os.path.join(output_dir, 'Vocals_Reverb.wav')
    lead_vocals_no_reverb_path = os.path.join(output_dir, 'Vocals_No_Reverb.wav')
    lead_vocals_noise_path = os.path.join(output_dir, 'Vocals_Noise.wav')
    lead_vocals_no_noise_path = os.path.join(output_dir, 'Vocals_No_Noise.wav')
    
    # Step 1: Separate Vocals and Instrumental
    display_progress('[~] Separating Vocals from Instrumental...', 0.1, is_webui, progress)
    separator.load_model(model_filename='Kim_Vocal_1.onnx')
    voc_inst = separator.separate(song_input)
    
    # Rename files to standard names
    if os.path.exists(os.path.join(output_dir, voc_inst[0])):
        os.rename(os.path.join(output_dir, voc_inst[0]), instrumental_path)
    if os.path.exists(os.path.join(output_dir, voc_inst[1])):
        os.rename(os.path.join(output_dir, voc_inst[1]), vocals_path)
    
    # Step 2: Separate Lead Vocals from Backing Vocals
    display_progress('[~] Separating Main Vocals from Backup Vocals...', 0.2, is_webui, progress)
    separator.load_model(model_filename='UVR_MDXNET_KARA.onnx')
    backing_voc = separator.separate(vocals_path)
    
    # Rename files to standard names
    if os.path.exists(os.path.join(output_dir, backing_voc[0])):
        os.rename(os.path.join(output_dir, backing_voc[0]), backing_vocals_path)
    if os.path.exists(os.path.join(output_dir, backing_voc[1])):
        os.rename(os.path.join(output_dir, backing_voc[1]), lead_vocals_path)
    
    # Step 3: Apply DeReverb to Lead Vocals
    display_progress('[~] Applying DeReverb to Vocals...', 0.3, is_webui, progress)
    separator.load_model(model_filename='UVR-De-Echo-Aggressive.pth')
    voc_no_reverb = separator.separate(lead_vocals_path)
    
    # Rename files to standard names
    if os.path.exists(os.path.join(output_dir, voc_no_reverb[0])):
        os.rename(os.path.join(output_dir, voc_no_reverb[0]), lead_vocals_no_reverb_path)
    if os.path.exists(os.path.join(output_dir, voc_no_reverb[1])):
        os.rename(os.path.join(output_dir, voc_no_reverb[1]), lead_vocals_reverb_path)
    
    # Step 4: Apply Denoise to Lead Vocals
    display_progress('[~] Applying Denoise to Vocals...', 0.4, is_webui, progress)
    separator.load_model(model_filename='UVR-DeNoise.pth')
    voc_no_noise = separator.separate(lead_vocals_no_reverb_path)
    
    # Rename files to standard names
    if os.path.exists(os.path.join(output_dir, voc_no_noise[0])):
        os.rename(os.path.join(output_dir, voc_no_noise[0]), lead_vocals_noise_path)
    if os.path.exists(os.path.join(output_dir, voc_no_noise[1])):
        os.rename(os.path.join(output_dir, voc_no_noise[1]), lead_vocals_no_noise_path)
    
    # Rename files to match the expected format from main.py
    # main.py expects files with specific suffixes
    base_name = os.path.splitext(os.path.basename(song_input))[0]
    
    # Rename to match main.py expectations
    main_vocals_renamed = os.path.join(output_dir, f"{base_name}_Vocals_Main.wav")
    backup_vocals_renamed = os.path.join(output_dir, f"{base_name}_Vocals_Backup.wav")
    instrumentals_renamed = os.path.join(output_dir, f"{base_name}_Instrumental.wav")
    vocals_renamed = os.path.join(output_dir, f"{base_name}_Vocals.wav")
    main_vocals_dereverb_renamed = os.path.join(output_dir, f"{base_name}_Vocals_Main_DeReverb.wav")
    
    if os.path.exists(lead_vocals_path):
        os.rename(lead_vocals_path, main_vocals_renamed)
    if os.path.exists(backing_vocals_path):
        os.rename(backing_vocals_path, backup_vocals_renamed)
    if os.path.exists(instrumental_path):
        os.rename(instrumental_path, instrumentals_renamed)
    if os.path.exists(vocals_path):
        os.rename(vocals_path, vocals_renamed)
    if os.path.exists(lead_vocals_no_noise_path):
        os.rename(lead_vocals_no_noise_path, main_vocals_dereverb_renamed)
    
    # Return the paths that match the expected format from main.py
    # main.py expects: orig_song_path, vocals_path, instrumentals_path, main_vocals_path, backup_vocals_path, main_vocals_dereverb_path
    return (
        song_input,  # orig_song_path
        vocals_renamed,  # vocals_path
        instrumentals_renamed,  # instrumentals_path  
        main_vocals_renamed,  # main_vocals_path
        backup_vocals_renamed,  # backup_vocals_path
        main_vocals_dereverb_renamed  # main_vocals_dereverb_path
    )


def preprocess_song_uvr(song_input, song_id, is_webui, input_type, progress=None):
    """
    Preprocess song using UVR separation instead of MDX
    
    This function replaces the preprocess_song function in main.py
    but uses UVR separation pipeline instead of MDX
    """
    # Import functions from main.py
    import main
    
    keep_orig = False
    if input_type == 'yt':
        display_progress('[~] Downloading song...', 0, is_webui, progress)
        song_link = song_input.split('&')[0]
        orig_song_path = main.yt_download(song_link)
    elif input_type == 'local':
        orig_song_path = song_input
        keep_orig = True
    else:
        orig_song_path = None

    song_output_dir = os.path.join(main.output_dir, song_id)
    orig_song_path = main.convert_to_stereo(orig_song_path)

    # Use UVR separation instead of MDX
    orig_song_path, vocals_path, instrumentals_path, main_vocals_path, backup_vocals_path, main_vocals_dereverb_path = run_uvr_separation(
        orig_song_path, song_output_dir, song_id, is_webui, input_type, progress
    )

    return orig_song_path, vocals_path, instrumentals_path, main_vocals_path, backup_vocals_path, main_vocals_dereverb_path


 