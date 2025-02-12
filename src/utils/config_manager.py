"""Module de gestion de la configuration"""

import yaml
import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from .error_handler import setup_logger

class ConfigManager:
    """Gestionnaire de configuration avec validation et valeurs par défaut"""
    
    DEFAULT_CONFIG = {
        'chrome': {
            'window_size': {
                'width': 1920,
                'height': 1080
            },
            'options': [
                '--no-sandbox',
                '--disable-dev-shm-usage',
                '--disable-blink-features=AutomationControlled',
                '--disable-extensions',
                '--disable-notifications',
                '--start-maximized',
                '--lang=en-US,en;q=0.9'
            ],
            'user_agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0'
            ),
            'prefs': {
                'profile.default_content_setting_values.notifications': 2,
                'credentials_enable_service': False,
                'profile.password_manager_enabled': False,
                'profile.default_content_settings.popups': 0,
                'download.prompt_for_download': False
            }
        },
        'output_dir': 'data',
        'max_results': 1000,
        'export_format': 'excel',
        'rate_limit_delay': 2.0,
        'max_retries': 3,
        'use_proxy': False,
        'log_level': 'INFO',
        'slr_config': {
            'analysis': {
                'temporal': {
                    'window_size': 5,
                    'min_citations': 5,
                    'growth_metrics': True
                },
                'topics': {
                    'num_topics': 10,
                    'max_features': 2000,
                    'update_interval': 1
                },
                'citations': {
                    'min_citations': 1,
                    'network_layout': 'spring',
                    'include_self_citations': False
                },
                'quality': {
                    'methodology_keywords': {
                        'empirical': [
                            'experiment',
                            'case study',
                            'survey',
                            'questionnaire'
                        ],
                        'theoretical': [
                            'theory',
                            'framework',
                            'model',
                            'concept'
                        ],
                        'validation': [
                            'validation',
                            'evaluation',
                            'assessment',
                            'testing'
                        ]
                    }
                },
                'text_processing': {
                    'language': 'english',
                    'remove_stopwords': True,
                    'lemmatize': True,
                    'min_word_length': 3,
                    'custom_stopwords': []
                }
            },
            'visualization': {
                'style': 'plotly_white',
                'colorscale': 'Viridis',
                'font_family': 'Arial',
                'temporal_plot': {
                    'include_trendline': True,
                    'show_confidence_interval': True
                },
                'network_plot': {
                    'node_size_factor': 10,
                    'edge_width': 0.5,
                    'show_labels': True
                },
                'topic_plot': {
                    'top_n_words': 10,
                    'show_word_cloud': True
                },
                'methodology_plot': {
                    'plot_type': 'bar',
                    'show_percentages': True
                }
            },
            'export': {
                'formats': ['excel', 'pdf', 'html'],
                'pdf': {
                    'paper_size': 'A4',
                    'orientation': 'portrait',
                    'font_family': 'Arial',
                    'include_figures': True,
                    'dpi': 300
                },
                'excel': {
                    'include_metadata': True,
                    'separate_sheets': True,
                    'sheet_names': [
                        'Summary',
                        'Papers',
                        'Authors',
                        'Topics',
                        'Citations'
                    ]
                },
                'html': {
                    'template': 'academic',
                    'interactive': True,
                    'include_plotly': True
                }
            }
        }
    }
    
    def __init__(self, config_path: str):
        """Initialise le gestionnaire de configuration"""
        self.logger = setup_logger("config_manager")
        
        try:
            # Validation du chemin
            if not isinstance(config_path, str):
                raise ValueError(f"Config path must be a string, got {type(config_path)}")
            
            # Conversion en chemin absolu
            config_path = os.path.abspath(config_path)
            self.logger.info(f"Using config path: {config_path}")
            
            self.config_path = config_path
            self.config = self._load_config()
            
        except Exception as e:
            self.logger.error(f"Error initializing config manager: {str(e)}")
            self.config_path = None
            self.config = self.DEFAULT_CONFIG.copy()
            
    def _load_config(self) -> Dict[str, Any]:
        """Charge et valide la configuration"""
        try:
            # Charger le fichier de configuration
            if self.config_path and os.path.exists(self.config_path):
                self.logger.info(f"Loading config from: {self.config_path}")
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    user_config = yaml.safe_load(f)
                    if user_config is None:
                        self.logger.warning("Empty config file, using defaults")
                        user_config = {}
            else:
                self.logger.warning(f"Config file not found at {self.config_path}, using defaults")
                user_config = {}
            
            # Fusionner avec les valeurs par défaut
            config = self._merge_configs(self.DEFAULT_CONFIG, user_config)
            
            # Valider la configuration
            self._validate_config(config)
            
            return config
            
        except Exception as e:
            self.logger.error(f"Error loading config: {str(e)}")
            self.logger.info("Falling back to default configuration")
            return self.DEFAULT_CONFIG.copy()
    
    def _merge_configs(self, default: Dict, user: Dict) -> Dict:
        """Fusionne récursivement la configuration utilisateur avec les valeurs par défaut"""
        merged = default.copy()
        
        for key, value in user.items():
            if (
                key in merged and 
                isinstance(merged[key], dict) and 
                isinstance(value, dict)
            ):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def _validate_config(self, config: Dict) -> None:
        """Valide la configuration et ses valeurs"""
        # Vérifier les valeurs requises
        required_keys = ['chrome', 'output_dir', 'max_results']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
        
        # Valider les valeurs numériques
        if config['max_results'] <= 0:
            raise ValueError("max_results must be positive")
        
        if config['rate_limit_delay'] < 0:
            raise ValueError("rate_limit_delay cannot be negative")
        
        # Valider les chemins
        output_dir = Path(config['output_dir'])
        if not output_dir.exists():
            self.logger.info(f"Creating output directory: {output_dir}")
            output_dir.mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Récupère une valeur de configuration avec gestion des clés imbriquées"""
        try:
            value = self.config
            for k in key.split('.'):
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """Définit une valeur de configuration"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self) -> None:
        """Sauvegarde la configuration dans le fichier"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            self.logger.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            self.logger.error(f"Error saving config: {str(e)}")
            raise
    
    def reset_to_defaults(self) -> None:
        """Réinitialise la configuration aux valeurs par défaut"""
        self.config = self.DEFAULT_CONFIG.copy()
        self.save()
        self.logger.info("Configuration reset to defaults")
    
    def validate_path(self, path: str) -> bool:
        """Valide un chemin de fichier ou dossier"""
        try:
            Path(path).resolve()
            return True
        except (TypeError, ValueError):
            return False
    
    def get_chrome_options(self) -> Dict[str, Any]:
        """Récupère les options Chrome avec validation"""
        chrome_config = self.get('chrome', {})
        
        # Valider la taille de la fenêtre
        window_size = chrome_config.get('window_size', {})
        if not isinstance(window_size.get('width'), int) or not isinstance(window_size.get('height'), int):
            self.logger.warning("Invalid window size, using defaults")
            window_size = self.DEFAULT_CONFIG['chrome']['window_size']
        
        # Valider les options
        options = chrome_config.get('options', [])
        if not isinstance(options, list):
            self.logger.warning("Invalid chrome options, using defaults")
            options = self.DEFAULT_CONFIG['chrome']['options']
        
        return {
            'window_size': window_size,
            'options': options,
            'user_agent': chrome_config.get('user_agent', self.DEFAULT_CONFIG['chrome']['user_agent'])
        }
