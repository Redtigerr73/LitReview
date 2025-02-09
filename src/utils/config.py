"""
Configuration management for LitReview
"""

import os
import yaml
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    """Configuration settings for LitReview"""
    output_dir: str
    max_results: int
    export_format: str
    rate_limit_delay: float
    log_level: str
    
    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """Load configuration from YAML file"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found: {path}")
            
        with open(path, 'r') as f:
            config_data = yaml.safe_load(f)
            
        return cls(
            output_dir=config_data.get('output_dir', 'data'),
            max_results=config_data.get('max_results', 1000),
            export_format=config_data.get('export_format', 'excel'),
            rate_limit_delay=config_data.get('rate_limit_delay', 2.0),
            log_level=config_data.get('log_level', 'INFO')
        )
        
    @classmethod
    def default(cls) -> 'Config':
        """Create default configuration"""
        return cls(
            output_dir='data',
            max_results=1000,
            export_format='excel',
            rate_limit_delay=2.0,
            log_level='INFO'
        )
