""" This file contains the Separator class, to facilitate the separation of stems from audio. """

import os
import sys
import platform
import subprocess
import time
import logging
import warnings
import importlib
import io
from typing import Optional

import hashlib
import json
import yaml
import requests
import torch
import torch.amp.autocast_mode as autocast_mode
import onnxruntime as ort
from tqdm import tqdm

# Import local architecture modules
from .architectures.mdx_separator import MDXSeparator
from .architectures.vr_separator import VRSeparator
from .architectures.demucs_separator import DemucsSeparator
from .architectures.mdxc_separator import MDXCSeparator


class Separator:
    """
    The Separator class is designed to facilitate the separation of audio sources from a given audio file.
    It supports various separation architectures and models, including MDX, VR, and Demucs. The class provides
    functionalities to configure separation parameters, load models, and perform audio source separation.
    It also handles logging, normalization, and output formatting of the separated audio stems.

    The actual separation task is handled by one of the architecture-specific classes in the `architectures` module;
    this class is responsible for initialising logging, configuring hardware acceleration, loading the model,
    initiating the separation process and passing outputs back to the caller.

    Common Attributes:
        log_level (int): The logging level.
        log_formatter (logging.Formatter): The logging formatter.
        model_file_dir (str): The directory where model files are stored.
        output_dir (str): The directory where output files will be saved.
        output_format (str): The format of the output audio file.
        output_bitrate (str): The bitrate of the output audio file.
        amplification_threshold (float): The threshold for audio amplification.
        normalization_threshold (float): The threshold for audio normalization.
        output_single_stem (str): Option to output a single stem.
        invert_using_spec (bool): Flag to invert using spectrogram.
        sample_rate (int): The sample rate of the audio.
        use_soundfile (bool): Use soundfile for audio writing, can solve OOM issues.
        use_autocast (bool): Flag to use PyTorch autocast for faster inference.

    MDX Architecture Specific Attributes:
        hop_length (int): The hop length for STFT.
        segment_size (int): The segment size for processing.
        overlap (float): The overlap between segments.
        batch_size (int): The batch size for processing.
        enable_denoise (bool): Flag to enable or disable denoising.

    VR Architecture Specific Attributes & Defaults:
        batch_size: 16
        window_size: 512
        aggression: 5
        enable_tta: False
        enable_post_process: False
        post_process_threshold: 0.2
        high_end_process: False

    Demucs Architecture Specific Attributes & Defaults:
        segment_size: "Default"
        shifts: 2
        overlap: 0.25
        segments_enabled: True

    MDXC Architecture Specific Attributes & Defaults:
        segment_size: 256
        override_model_segment_size: False
        batch_size: 1
        overlap: 8
        pitch_shift: 0
    """

    def __init__(
        self,
        log_level=logging.INFO,
        log_formatter=None,
        model_file_dir="uvr_models/",
        output_dir=None,
        output_format="WAV",
        output_bitrate=None,
        normalization_threshold=0.9,
        amplification_threshold=0.0,
        output_single_stem=None,
        invert_using_spec=False,
        sample_rate=44100,
        use_soundfile=False,
        use_autocast=False,
        use_directml=False,
        mdx_params={"hop_length": 1024, "segment_size": 256, "overlap": 0.25, "batch_size": 1, "enable_denoise": False},
        vr_params={"batch_size": 1, "window_size": 512, "aggression": 5, "enable_tta": False, "enable_post_process": False, "post_process_threshold": 0.2, "high_end_process": False},
        demucs_params={"segment_size": "Default", "shifts": 2, "overlap": 0.25, "segments_enabled": True},
        mdxc_params={"segment_size": 256, "override_model_segment_size": False, "batch_size": 1, "overlap": 8, "pitch_shift": 0},
        info_only=False,
    ):
        """Initialize the separator."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.log_level = log_level
        self.log_formatter = log_formatter

        self.log_handler = logging.StreamHandler()

        if self.log_formatter is None:
            self.log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(module)s - %(message)s")

        self.log_handler.setFormatter(self.log_formatter)

        if not self.logger.hasHandlers():
            self.logger.addHandler(self.log_handler)

        # Filter out noisy warnings from PyTorch for users who don't care about them
        if log_level > logging.DEBUG:
            warnings.filterwarnings("ignore")

        # Skip initialization logs if info_only is True
        if not info_only:
            self.logger.info(f"AICoverGen UVR Separator instantiating with output_dir: {output_dir}, output_format: {output_format}")

        if output_dir is None:
            output_dir = os.getcwd()
            if not info_only:
                self.logger.info("Output directory not specified. Using current working directory.")

        self.output_dir = output_dir

        # Check for environment variable to override model_file_dir
        env_model_dir = os.environ.get("AUDIO_SEPARATOR_MODEL_DIR")
        if env_model_dir:
            self.model_file_dir = env_model_dir
            self.logger.info(f"Using model directory from AUDIO_SEPARATOR_MODEL_DIR env var: {self.model_file_dir}")
            if not os.path.exists(self.model_file_dir):
                raise FileNotFoundError(f"The specified model directory does not exist: {self.model_file_dir}")
        else:
            self.logger.info(f"Using model directory from model_file_dir parameter: {model_file_dir}")
            self.model_file_dir = model_file_dir

        # Create the model directory if it does not exist
        os.makedirs(self.model_file_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        self.output_format = output_format
        self.output_bitrate = output_bitrate

        if self.output_format is None:
            self.output_format = "WAV"

        self.normalization_threshold = normalization_threshold
        if normalization_threshold <= 0 or normalization_threshold > 1:
            raise ValueError("The normalization_threshold must be greater than 0 and less than or equal to 1.")

        self.amplification_threshold = amplification_threshold
        if amplification_threshold < 0 or amplification_threshold > 1:
            raise ValueError("The amplification_threshold must be greater than or equal to 0 and less than or equal to 1.")

        self.output_single_stem = output_single_stem
        if self.output_single_stem is not None:
            self.logger.debug(f"Single stem output requested, so only one output file ({self.output_single_stem}) will be written")

        self.invert_using_spec = invert_using_spec
        if self.invert_using_spec:
            self.logger.debug(f"Secondary step will be inverted using spectogram rather than waveform. This may improve quality but is slightly slower.")

        try:
            self.sample_rate = int(sample_rate)
            if self.sample_rate <= 0:
                raise ValueError(f"The sample rate setting is {self.sample_rate} but it must be a non-zero whole number.")
            if self.sample_rate > 12800000:
                raise ValueError(f"The sample rate setting is {self.sample_rate}. Enter something less ambitious.")
        except ValueError:
            raise ValueError("The sample rate must be a non-zero whole number. Please provide a valid integer.")

        self.use_soundfile = use_soundfile
        self.use_autocast = use_autocast
        self.use_directml = use_directml

        # These are parameters which users may want to configure so we expose them to the top-level Separator class,
        # even though they are specific to a single model architecture
        self.arch_specific_params = {"MDX": mdx_params, "VR": vr_params, "Demucs": demucs_params, "MDXC": mdxc_params}

        self.torch_device = None
        self.torch_device_cpu = None
        self.torch_device_mps = None

        self.onnx_execution_provider = None
        self.model_instance = None

        self.model_is_uvr_vip = False
        self.model_friendly_name = None

        if not info_only:
            self.setup_accelerated_inferencing_device()

    def setup_accelerated_inferencing_device(self):
        """Setup the best available inference device."""
        system_info = self.get_system_info()
        self.logger.info(f"System: {system_info}")

        # Setup ONNX Runtime providers
        ort_providers = ['CPUExecutionProvider']

        if system_info['cuda_available']:
            self.configure_cuda(ort_providers)
        elif system_info['mps_available']:
            self.configure_mps(ort_providers)
        elif system_info['directml_available'] and self.use_directml:
            self.configure_dml(ort_providers)

        self.onnx_execution_provider = ort_providers[0]
        self.logger.info(f"Using ONNX Runtime provider: {self.onnx_execution_provider}")

        # Setup PyTorch device
        self.setup_torch_device(system_info)

    def get_system_info(self):
        """Get system information for device selection."""
        return {
            'platform': platform.system(),
            'cuda_available': torch.cuda.is_available(),
            'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
            'directml_available': hasattr(torch.backends, 'directml') and torch.backends.directml.is_available(),
            'cpu_count': os.cpu_count()
        }

    def check_ffmpeg_installed(self):
        """Check if ffmpeg is installed."""
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def log_onnxruntime_packages(self):
        """Log ONNX Runtime package information."""
        self.logger.info(f"ONNX Runtime version: {ort.__version__}")
        providers = ort.get_available_providers()
        self.logger.info(f"Available ONNX Runtime providers: {providers}")

    def setup_torch_device(self, system_info):
        """Setup PyTorch device."""
        if system_info['cuda_available']:
            self.torch_device = torch.device('cuda')
            self.torch_device_cpu = torch.device('cpu')
            self.logger.info(f"PyTorch CUDA device: {self.torch_device}")
        elif system_info['mps_available']:
            self.torch_device = torch.device('mps')
            self.torch_device_cpu = torch.device('cpu')
            self.logger.info(f"PyTorch MPS device: {self.torch_device}")
        else:
            self.torch_device = torch.device('cpu')
            self.torch_device_cpu = torch.device('cpu')
            self.logger.info(f"PyTorch CPU device: {self.torch_device}")

    def configure_cuda(self, ort_providers):
        """Configure CUDA for ONNX Runtime."""
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            ort_providers.insert(0, 'CUDAExecutionProvider')
            self.logger.info("CUDA provider configured for ONNX Runtime")

    def configure_mps(self, ort_providers):
        """Configure MPS for ONNX Runtime."""
        if 'CoreMLExecutionProvider' in ort.get_available_providers():
            ort_providers.insert(0, 'CoreMLExecutionProvider')
            self.logger.info("CoreML provider configured for ONNX Runtime")

    def configure_dml(self, ort_providers):
        """Configure DirectML for ONNX Runtime."""
        if 'DmlExecutionProvider' in ort.get_available_providers():
            ort_providers.insert(0, 'DmlExecutionProvider')
            self.logger.info("DirectML provider configured for ONNX Runtime")

    def get_model_hash(self, model_path):
        """Get hash of model file."""
        try:
            with open(model_path, 'rb') as f:
                f.seek(-10000 * 1024, 2)
                model_hash = hashlib.md5(f.read()).hexdigest()
        except:
            model_hash = hashlib.md5(open(model_path, 'rb').read()).hexdigest()
        return model_hash

    def download_file_if_not_exists(self, url, output_path):
        """Download file if it doesn't exist."""
        if os.path.exists(output_path):
            self.logger.info(f"File already exists: {output_path}")
            return

        self.logger.info(f"Downloading {url} to {output_path}")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            self.logger.info(f"Download completed: {output_path}")
        except Exception as e:
            self.logger.error(f"Download failed: {e}")
            raise

    def load_model(self, model_filename="model_bs_roformer_ep_317_sdr_12.9755.ckpt"):
        """Load a separation model."""
        model_path = os.path.join(self.model_file_dir, model_filename)
        
        if not os.path.exists(model_path):
            self.logger.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # 모델 메타데이터 로드
        with open(os.path.join(self.model_file_dir, "model_data.json")) as f:
            model_data = json.load(f)

        common_config = {
            'model_path': model_path,
            'device': self.torch_device,
            'use_autocast': self.use_autocast,
            'output_dir': self.output_dir,
            'output_format': self.output_format,
            'normalization_threshold': self.normalization_threshold,
            'amplification_threshold': self.amplification_threshold,
            'output_single_stem': self.output_single_stem,
            'invert_using_spec': self.invert_using_spec,
            'sample_rate': self.sample_rate,
            'use_soundfile': self.use_soundfile,
            'model_data': model_data,
            'logger': self.logger,  # ← 반드시 추가!
            'log_level': self.log_level,  # ← (필요시)
        }
        arch_config = self.arch_specific_params["MDX"]
        self.model_instance = MDXSeparator(common_config, arch_config)

        self.logger.info(f"Model loaded successfully: {model_filename}")

    def separate(self, audio_file_path, custom_output_names=None):
        """Separate audio file into stems."""
        if self.model_instance is None:
            raise ValueError("No model loaded. Call load_model() first.")

        self.logger.info(f"Starting separation of: {audio_file_path}")
        
        # Perform separation using the loaded model
        output_files = self.model_instance.separate(
            audio_file_path,
            custom_output_names=custom_output_names
        )

        self.logger.info(f"Separation completed. Output files: {output_files}")
        return output_files 