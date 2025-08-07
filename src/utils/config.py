"""Configuration utilities for the content performance predictor."""

import os
import yaml
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Configuration manager for the application."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file and environment variables."""
        config = {}
        
        # Load YAML config if exists
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
        
        # Override with environment variables
        config = self._override_with_env(config)
        
        return config
    
    def _override_with_env(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Override configuration with environment variables."""
        # Data configuration
        if 'data' not in config:
            config['data'] = {}
        
        config['data']['supabase'] = {
            'url': os.getenv('SUPABASE_URL', config.get('data', {}).get('supabase', {}).get('url')),
            'key': os.getenv('SUPABASE_KEY', config.get('data', {}).get('supabase', {}).get('key')),
            'service_role_key': os.getenv('SUPABASE_SERVICE_ROLE_KEY', config.get('data', {}).get('supabase', {}).get('service_role_key'))
        }
        
        # API configuration
        if 'api' not in config:
            config['api'] = {}
        
        config['api'].update({
            'host': os.getenv('API_HOST', config.get('api', {}).get('host', '0.0.0.0')),
            'port': int(os.getenv('API_PORT', config.get('api', {}).get('port', 8000))),
            'reload': os.getenv('API_RELOAD', config.get('api', {}).get('reload', 'true')).lower() == 'true'
        })
        
        # Dashboard configuration
        if 'dashboard' not in config:
            config['dashboard'] = {}
        
        config['dashboard'].update({
            'host': os.getenv('DASHBOARD_HOST', config.get('dashboard', {}).get('host', '0.0.0.0')),
            'port': int(os.getenv('DASHBOARD_PORT', config.get('dashboard', {}).get('port', 8050))),
            'debug': os.getenv('DASHBOARD_DEBUG', config.get('dashboard', {}).get('debug', 'true')).lower() == 'true'
        })
        
        # MLflow configuration
        if 'mlflow' not in config:
            config['mlflow'] = {}
        
        config['mlflow'].update({
            'tracking_uri': os.getenv('MLFLOW_TRACKING_URI', config.get('mlflow', {}).get('tracking_uri', 'http://localhost:5000')),
            'experiment_name': os.getenv('MLFLOW_EXPERIMENT_NAME', config.get('mlflow', {}).get('experiment_name', 'content-performance-predictor'))
        })
        
        # Paths
        if 'paths' not in config:
            config['paths'] = {}
        
        config['paths'].update({
            'models': os.getenv('MODEL_PATH', config.get('paths', {}).get('models', './models')),
            'data': os.getenv('DATA_PATH', config.get('paths', {}).get('data', './data')),
            'features': os.getenv('FEATURE_STORE_PATH', config.get('paths', {}).get('features', './data/features')),
            'experiments': os.getenv('EXPERIMENT_PATH', config.get('paths', {}).get('experiments', './mlflow')),
            'logs': os.getenv('LOG_PATH', config.get('paths', {}).get('logs', './logs'))
        })
        
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key (dot notation supported)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_supabase_config(self) -> Dict[str, str]:
        """Get Supabase configuration."""
        return self.get('data.supabase', {})
    
    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration."""
        return self.get('api', {})
    
    def get_dashboard_config(self) -> Dict[str, Any]:
        """Get dashboard configuration."""
        return self.get('dashboard', {})
    
    def get_mlflow_config(self) -> Dict[str, str]:
        """Get MLflow configuration."""
        return self.get('mlflow', {})
    
    def get_paths(self) -> Dict[str, str]:
        """Get paths configuration."""
        return self.get('paths', {})
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.get('models', {})
    
    def get_feature_config(self) -> Dict[str, Any]:
        """Get feature engineering configuration."""
        return self.get('features', {})


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get global configuration instance."""
    return config