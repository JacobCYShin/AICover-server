"""Module for separating audio sources using MDX architecture models."""

import os
import platform
import torch
import onnx
import onnxruntime as ort
import numpy as np
import onnx2torch
from tqdm import tqdm
import librosa
import soundfile as sf
from onnx import version_converter

# Import local modules
from ..uvr_lib_v5 import spec_utils
from ..uvr_lib_v5.stft import STFT
from .common_separator import CommonSeparator


class MDXSeparator(CommonSeparator):
    """
    MDXSeparator is responsible for separating audio sources using MDX models.
    It initializes with configuration parameters and prepares the model for separation tasks.
    """

    def check_and_upgrade_onnx_model(self, model_path):
        """
        Check ONNX model IR version and upgrade if necessary.
        
        Args:
            model_path (str): Path to the ONNX model file
            
        Returns:
            bool: True if model was upgraded, False if no upgrade needed
        """
        try:
            self.logger.info(f"Checking ONNX model IR version: {model_path}")
            
            # Load the model to check IR version
            model = onnx.load(model_path)
            current_ir_version = model.ir_version
            self.logger.info(f"Current ONNX IR version: {current_ir_version}")
            
            # Check if upgrade is needed (IR version < 3)
            if current_ir_version < 3:
                self.logger.warning(f"ONNX model IR version {current_ir_version} is too old (minimum supported: 3)")
                self.logger.info("Starting ONNX model upgrade...")
                
                try:
                    # Convert to IR version 7 (modern version)
                    target_version = 7
                    self.logger.info(f"Converting ONNX model to IR version {target_version}...")
                    converted_model = version_converter.convert_version(model, target_version)
                    
                    # Save the converted model back to the same path
                    onnx.save(converted_model, model_path)
                    
                    # Verify the conversion
                    verification_model = onnx.load(model_path)
                    new_ir_version = verification_model.ir_version
                    self.logger.info(f"ONNX model successfully upgraded to IR version {new_ir_version}")
                    
                    return True
                    
                except Exception as e:
                    self.logger.error(f"Failed to upgrade ONNX model: {e}")
                    self.logger.warning("Keeping original model file (upgrade failed)")
                    return False
            else:
                self.logger.info(f"ONNX model IR version {current_ir_version} is compatible (>= 3)")
                return False
                
        except Exception as e:
            self.logger.error(f"Error checking ONNX model version: {e}")
            return False

    def __init__(self, model_path, device, use_autocast=False, **arch_config):
        """
        Initialize MDXSeparator with model path and configuration.
        
        Args:
            model_path (str): Path to the ONNX model file
            device (torch.device): PyTorch device for inference
            use_autocast (bool): Whether to use autocast for mixed precision
            **arch_config: Architecture-specific configuration parameters
        """
        # Initialize common configuration
        common_config = {
            'model_path': model_path,
            'device': device,
            'use_autocast': use_autocast,
            'output_dir': arch_config.get('output_dir', './output'),
            'output_format': arch_config.get('output_format', 'WAV'),
            'normalization_threshold': arch_config.get('normalization_threshold', 0.9),
            'amplification_threshold': arch_config.get('amplification_threshold', 0.0),
            'sample_rate': arch_config.get('sample_rate', 44100),
            'invert_using_spec': arch_config.get('invert_using_spec', False),
            'output_single_stem': arch_config.get('output_single_stem', None)
        }
        
        super().__init__(config=common_config)

        # MDX-specific configuration
        self.segment_size = arch_config.get("segment_size", 256)
        self.overlap = arch_config.get("overlap", 0.25)
        self.batch_size = arch_config.get("batch_size", 1)
        self.hop_length = arch_config.get("hop_length", 1024)
        self.enable_denoise = arch_config.get("enable_denoise", False)

        # Load model data from the model file
        self.load_model_data()
        
        # Initialize model-specific parameters
        self.compensate = self.model_data.get("compensate", 1.0)
        self.dim_f = self.model_data.get("mdx_dim_f_set", 3072)
        self.dim_t = 2 ** self.model_data.get("mdx_dim_t_set", 9)
        self.n_fft = self.model_data.get("mdx_n_fft_scale_set", 2048)
        self.config_yaml = self.model_data.get("config_yaml", None)

        # Initialize model
        self.load_model()

        # Initialize processing parameters
        self.n_bins = 0
        self.trim = 0
        self.chunk_size = 0
        self.gen_size = 0
        self.stft = None

        self.primary_source = None
        self.secondary_source = None
        self.audio_file_path = None
        self.audio_file_base = None

    def load_model_data(self):
        """Load model data from JSON file or create default data."""
        # For now, create default model data
        # In a full implementation, this would load from a JSON file
        self.model_data = {
            "compensate": 1.0,
            "mdx_dim_f_set": 3072,
            "mdx_dim_t_set": 9,
            "mdx_n_fft_scale_set": 2048,
            "primary_stem": "Vocals",
            "secondary_stem": "Instrumental"
        }

    def load_model(self):
        """
        Load the model into memory from file on disk, initialize it with config from the model data,
        and prepare for inferencing using hardware accelerated Torch device.
        """
        self.logger.debug("Loading ONNX model for inference...")
        
        # Check and upgrade ONNX model if necessary
        model_upgraded = self.check_and_upgrade_onnx_model(self.model_path)
        if model_upgraded:
            self.logger.info("ONNX model was upgraded, proceeding with loading...")
        else:
            self.logger.info("ONNX model version check completed, proceeding with loading...")

        if self.segment_size == self.dim_t:
            ort_session_options = ort.SessionOptions()
            ort_session_options.log_severity_level = 0

            ort_inference_session = ort.InferenceSession(
                self.model_path, 
                providers=['CPUExecutionProvider'], 
                sess_options=ort_session_options
            )
            self.model_run = lambda spek: ort_inference_session.run(None, {"input": spek.cpu().numpy()})[0]
            self.logger.debug("Model loaded successfully using ONNXruntime inferencing session.")
        else:
            if platform.system() == 'Windows':
                onnx_model = onnx.load(self.model_path)
                self.model_run = onnx2torch.convert(onnx_model)
            else:
                self.model_run = onnx2torch.convert(self.model_path)
   
            self.model_run.to(self.device).eval()
            self.logger.warning("Model converted from onnx to pytorch due to segment size not matching dim_t, processing may be slower.")

    def separate(self, audio_file_path, custom_output_names=None):
        """
        Separates the audio file into primary and secondary sources based on the model's configuration.
        It processes the mix, demixes it into sources, normalizes the sources, and saves the output files.

        Args:
            audio_file_path (str): The path to the audio file to be processed.
            custom_output_names (dict, optional): Custom names for the output files. Defaults to None.

        Returns:
            list: A list of paths to the output files generated by the separation process.
        """
        self.audio_file_path = audio_file_path
        self.audio_file_base = os.path.splitext(os.path.basename(audio_file_path))[0]

        # Prepare the mix for processing
        self.logger.debug(f"Preparing mix for input audio file {self.audio_file_path}...")
        mix = self.prepare_mix(self.audio_file_path)

        self.logger.debug("Normalizing mix before demixing...")
        mix = spec_utils.normalize(wave=mix, max_peak=self.normalization_threshold, min_peak=self.amplification_threshold)

        # Start the demixing process
        source = self.demix(mix)
        self.logger.debug("Demixing completed.")

        # Initialize the list for output files
        output_files = []
        self.logger.debug("Processing output files...")

        # Normalize and transpose the primary source if it's not already an array
        if not isinstance(self.primary_source, np.ndarray):
            self.logger.debug("Normalizing primary source...")
            self.primary_source = spec_utils.normalize(wave=source, max_peak=self.normalization_threshold, min_peak=self.amplification_threshold).T

        # Process the secondary source if not already an array
        if not isinstance(self.secondary_source, np.ndarray):
            self.logger.debug("Producing secondary source: demixing in match_mix mode")
            raw_mix = self.demix(mix, is_match_mix=True)

            if self.invert_using_spec:
                self.logger.debug("Inverting secondary stem using spectogram as invert_using_spec is set to True")
                self.secondary_source = spec_utils.invert_stem(raw_mix, source)
            else:
                self.logger.debug("Inverting secondary stem by subtracting of transposed demixed stem from transposed original mix")
                self.secondary_source = mix.T - source.T

        # Save and process the secondary stem if needed
        secondary_stem_name = self.model_data.get("secondary_stem", "Instrumental")
        if not self.output_single_stem or self.output_single_stem.lower() == secondary_stem_name.lower():
            secondary_stem_output_path = self.get_stem_output_path(secondary_stem_name, custom_output_names)

            self.logger.info(f"Saving {secondary_stem_name} stem to {secondary_stem_output_path}...")
            self.final_process(secondary_stem_output_path, self.secondary_source, secondary_stem_name)
            output_files.append(secondary_stem_output_path)

        # Save and process the primary stem if needed
        primary_stem_name = self.model_data.get("primary_stem", "Vocals")
        if not self.output_single_stem or self.output_single_stem.lower() == primary_stem_name.lower():
            primary_stem_output_path = self.get_stem_output_path(primary_stem_name, custom_output_names)

            if not isinstance(self.primary_source, np.ndarray):
                self.primary_source = source.T

            self.logger.info(f"Saving {primary_stem_name} stem to {primary_stem_output_path}...")
            self.final_process(primary_stem_output_path, self.primary_source, primary_stem_name)
            output_files.append(primary_stem_output_path)

        return output_files

    def prepare_mix(self, audio_file_path):
        """Prepare the mix for processing."""
        # Load audio file
        mix, sr = librosa.load(audio_file_path, sr=self.sample_rate, mono=False)
        
        # Convert to mono if stereo
        if len(mix.shape) > 1:
            mix = np.mean(mix, axis=0)
        
        return mix

    def get_stem_output_path(self, stem_name, custom_output_names=None):
        """Get the output path for a stem."""
        if custom_output_names and stem_name in custom_output_names:
            filename = custom_output_names[stem_name]
        else:
            filename = f"{self.audio_file_base}_{stem_name}.{self.output_format.lower()}"
        
        return os.path.join(self.output_dir, filename)

    def final_process(self, output_path, source, stem_name):
        """Final processing and saving of the stem."""
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save the audio file
        sf.write(output_path, source, self.sample_rate)

    def initialize_model_settings(self):
        """Initialize model settings for processing."""
        self.n_bins = self.n_fft // 2 + 1
        self.trim = self.n_fft // 2
        self.chunk_size = self.hop_length * (self.dim_t - 1)
        self.gen_size = self.chunk_size - 2 * self.trim

    def initialize_mix(self, mix, is_ckpt=False):
        """Initialize the mix for processing."""
        self.initialize_model_settings()
        
        n_sample = mix.shape[1]
        pad = self.gen_size - n_sample % self.gen_size
        
        # Pad the mix
        mix_p = np.concatenate((np.zeros((2, self.trim)), mix, np.zeros((2, pad)), np.zeros((2, self.trim))), 1)
        
        mix_waves = []
        for i in range(0, n_sample + pad, self.gen_size):
            waves = np.array(mix_p[:, i:i + self.chunk_size])
            mix_waves.append(waves)
        
        return torch.tensor(mix_waves, dtype=torch.float32).to(self.device), pad, self.trim

    def demix(self, mix, is_match_mix=False):
        """Demix the audio into sources."""
        mix_waves, pad, trim = self.initialize_mix(mix)
        
        # Process the mix
        source = self.run_model(mix_waves, is_match_mix)
        
        # Remove padding
        source = source[:, :, trim:-trim].transpose(0, 1).reshape(2, -1).cpu().numpy()
        source = source[:, :-pad]
        
        return source

    def run_model(self, mix, is_match_mix=False):
        """Run the model on the mix."""
        with torch.no_grad():
            if self.use_autocast:
                with torch.autocast(device_type=self.device.type):
                    source = self.model_run(mix)
            else:
                source = self.model_run(mix)
        
        return source 