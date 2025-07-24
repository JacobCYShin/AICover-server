"""
STFT (Short-Time Fourier Transform) utilities for UVR audio processing.
"""

import numpy as np
import torch
import torch.nn as nn


class STFT(nn.Module):
    """
    Short-Time Fourier Transform module for PyTorch.
    """
    
    def __init__(self, n_fft=2048, hop_length=512, win_length=None, window='hann'):
        """
        Initialize STFT module.
        
        Args:
            n_fft (int): FFT window size
            hop_length (int): Hop length
            win_length (int): Window length
            window (str): Window type
        """
        super().__init__()
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length if win_length is not None else n_fft
        self.window = window
        
        # Create window function
        if window == 'hann':
            self.register_buffer('window', torch.hann_window(self.win_length))
        else:
            self.register_buffer('window', torch.ones(self.win_length))
    
    def forward(self, x):
        """
        Compute STFT.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, samples)
            
        Returns:
            torch.Tensor: STFT result of shape (batch, channels, freq_bins, time_frames)
        """
        return torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            return_complex=True
        )
    
    def inverse(self, stft_result):
        """
        Compute inverse STFT.
        
        Args:
            stft_result (torch.Tensor): STFT result
            
        Returns:
            torch.Tensor: Reconstructed audio
        """
        return torch.istft(
            stft_result,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True
        )


def stft_torch(audio, n_fft=2048, hop_length=512, win_length=None, window='hann'):
    """
    Compute STFT using PyTorch.
    
    Args:
        audio (torch.Tensor): Input audio tensor
        n_fft (int): FFT window size
        hop_length (int): Hop length
        win_length (int): Window length
        window (str): Window type
        
    Returns:
        torch.Tensor: STFT result
    """
    if win_length is None:
        win_length = n_fft
    
    if window == 'hann':
        window = torch.hann_window(win_length, device=audio.device)
    else:
        window = torch.ones(win_length, device=audio.device)
    
    return torch.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        return_complex=True
    )


def istft_torch(stft_result, n_fft=2048, hop_length=512, win_length=None, window='hann'):
    """
    Compute inverse STFT using PyTorch.
    
    Args:
        stft_result (torch.Tensor): STFT result
        n_fft (int): FFT window size
        hop_length (int): Hop length
        win_length (int): Window length
        window (str): Window type
        
    Returns:
        torch.Tensor: Reconstructed audio
    """
    if win_length is None:
        win_length = n_fft
    
    if window == 'hann':
        window = torch.hann_window(win_length, device=stft_result.device)
    else:
        window = torch.ones(win_length, device=stft_result.device)
    
    return torch.istft(
        stft_result,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True
    ) 