"""Common separator class for all architecture types."""

import os
import logging
import torch


class CommonSeparator:
    """
    Common separator class that provides shared functionality for all architecture types.
    """

    def __init__(self, config):
        """
        Initialize the common separator with configuration.
        
        Args:
            config (dict): Configuration dictionary containing common parameters
        """
        self.model_path = config.get('model_path')
        self.device = config.get('device', torch.device('cpu'))
        self.use_autocast = config.get('use_autocast', False)
        self.output_dir = config.get('output_dir', './output')
        self.output_format = config.get('output_format', 'WAV')
        self.normalization_threshold = config.get('normalization_threshold', 0.9)
        self.amplification_threshold = config.get('amplification_threshold', 0.0)
        self.sample_rate = config.get('sample_rate', 44100)
        self.invert_using_spec = config.get('invert_using_spec', False)
        self.output_single_stem = config.get('output_single_stem', None)
        
        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True) 