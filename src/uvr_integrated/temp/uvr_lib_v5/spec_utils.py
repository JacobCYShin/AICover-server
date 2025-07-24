"""
Spectrogram utilities for UVR audio processing.
"""

import numpy as np
import librosa
import soundfile as sf


def normalize(wave, max_peak=0.9, min_peak=0.0):
    """
    Normalize audio waveform.
    
    Args:
        wave (np.ndarray): Input waveform
        max_peak (float): Maximum peak value
        min_peak (float): Minimum peak value
        
    Returns:
        np.ndarray: Normalized waveform
    """
    if max_peak <= 0 or max_peak > 1:
        raise ValueError("max_peak must be between 0 and 1")
    
    if min_peak < 0 or min_peak > 1:
        raise ValueError("min_peak must be between 0 and 1")
    
    # Find the maximum absolute value
    max_val = np.max(np.abs(wave))
    
    if max_val == 0:
        return wave
    
    # Normalize to max_peak
    normalized = wave * (max_peak / max_val)
    
    # Apply minimum peak if needed
    if min_peak > 0:
        min_val = np.min(np.abs(normalized))
        if min_val < min_peak:
            normalized = normalized * (min_peak / min_val)
    
    return normalized


def invert_stem(raw_mix, source):
    """
    Invert a stem using spectrogram.
    
    Args:
        raw_mix (np.ndarray): Raw mix audio
        source (np.ndarray): Source audio to invert
        
    Returns:
        np.ndarray: Inverted stem
    """
    # Simple inversion by subtracting source from mix
    return raw_mix - source


def load_audio(file_path, sr=44100, mono=True):
    """
    Load audio file.
    
    Args:
        file_path (str): Path to audio file
        sr (int): Sample rate
        mono (bool): Convert to mono
        
    Returns:
        tuple: (audio_data, sample_rate)
    """
    audio, sr_orig = librosa.load(file_path, sr=sr, mono=mono)
    return audio, sr_orig


def save_audio(file_path, audio, sr=44100):
    """
    Save audio file.
    
    Args:
        file_path (str): Path to save audio file
        audio (np.ndarray): Audio data
        sr (int): Sample rate
    """
    sf.write(file_path, audio, sr)


def stft(audio, n_fft=2048, hop_length=512, win_length=None):
    """
    Compute Short-Time Fourier Transform.
    
    Args:
        audio (np.ndarray): Input audio
        n_fft (int): FFT window size
        hop_length (int): Hop length
        win_length (int): Window length
        
    Returns:
        np.ndarray: STFT result
    """
    if win_length is None:
        win_length = n_fft
    
    return librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def istft(stft_result, hop_length=512, win_length=None):
    """
    Compute Inverse Short-Time Fourier Transform.
    
    Args:
        stft_result (np.ndarray): STFT result
        hop_length (int): Hop length
        win_length (int): Window length
        
    Returns:
        np.ndarray: Reconstructed audio
    """
    if win_length is None:
        win_length = stft_result.shape[0] * 2 - 1
    
    return librosa.istft(stft_result, hop_length=hop_length, win_length=win_length)


def magnitude_phase(stft_result):
    """
    Extract magnitude and phase from STFT.
    
    Args:
        stft_result (np.ndarray): STFT result
        
    Returns:
        tuple: (magnitude, phase)
    """
    magnitude = np.abs(stft_result)
    phase = np.angle(stft_result)
    return magnitude, phase


def complex_from_magnitude_phase(magnitude, phase):
    """
    Reconstruct complex STFT from magnitude and phase.
    
    Args:
        magnitude (np.ndarray): Magnitude
        phase (np.ndarray): Phase
        
    Returns:
        np.ndarray: Complex STFT
    """
    return magnitude * np.exp(1j * phase) 