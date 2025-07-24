import os
import sys
import gc
import hashlib
import librosa
import numpy as np
import soundfile as sf
from urllib.parse import urlparse, parse_qs

# Import integrated UVR separator
from audio_separator.separator import Separator

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
    Run UVR separation pipeline using python-audio-separator (uvr_integrated)
    1. Vocals/Instrumental 분리
    2. Main/Backup Vocals 분리
    3. DeReverb
    4. Denoise
    """
    # Separator 인스턴스는 한 번만 생성
    separator = Separator(output_dir=output_dir)

    # 파일명 기본 세팅
    base_name = os.path.splitext(os.path.basename(song_input))[0]
    vocals_path = os.path.join(output_dir, f"{base_name}_Vocals.wav")
    instrumentals_path = os.path.join(output_dir, f"{base_name}_Instrumental.wav")
    main_vocals_path = os.path.join(output_dir, f"{base_name}_Vocals_Main.wav")
    backup_vocals_path = os.path.join(output_dir, f"{base_name}_Vocals_Backup.wav")
    main_vocals_dereverb_path = os.path.join(output_dir, f"{base_name}_Vocals_Main_DeReverb.wav")

    # 1. Vocals/Instrumental 분리
    display_progress('[~] Separating Vocals from Instrumental...', 0.1, is_webui, progress)
    separator.load_model(model_filename='Kim_Vocal_1.onnx')
    voc_inst = separator.separate(song_input)
    # voc_inst: [Instrumental, Vocals] (파일 경로 리스트)
    # 파일명 표준화
    if os.path.exists(voc_inst[0]):
        os.rename(voc_inst[0], instrumentals_path)
    if os.path.exists(voc_inst[1]):
        os.rename(voc_inst[1], vocals_path)

    # 2. Main/Backup Vocals 분리
    display_progress('[~] Separating Main Vocals from Backup Vocals...', 0.2, is_webui, progress)
    separator.load_model(model_filename='UVR_MDXNET_KARA.onnx')
    main_backup = separator.separate(vocals_path)
    # main_backup: [Backup, Main] (파일 경로 리스트)
    if os.path.exists(main_backup[0]):
        os.rename(main_backup[0], backup_vocals_path)
    if os.path.exists(main_backup[1]):
        os.rename(main_backup[1], main_vocals_path)

    # 3. DeReverb (Main Vocals)
    display_progress('[~] Applying DeReverb to Vocals...', 0.3, is_webui, progress)
    separator.load_model(model_filename='UVR-De-Echo-Aggressive.pth')
    dereverb = separator.separate(main_vocals_path)
    # dereverb: [NoReverb, Reverb] (파일 경로 리스트)
    if os.path.exists(dereverb[0]):
        os.rename(dereverb[0], main_vocals_dereverb_path)
    # 필요시 dereverb[1]도 활용 가능

    # 반환: main.py에서 기대하는 순서대로
    return (
        song_input,  # orig_song_path
        vocals_path,  # vocals_path
        instrumentals_path,  # instrumentals_path
        main_vocals_path,  # main_vocals_path
        backup_vocals_path,  # backup_vocals_path
        main_vocals_dereverb_path  # main_vocals_dereverb_path
    )


def preprocess_song_uvr(song_input, song_id, is_webui, input_type, progress=None):
    """
    Preprocess song using UVR separation instead of MDX
    """
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


 