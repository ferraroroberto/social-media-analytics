"""Configuration utilities for the content performance predictor."""

import os
import re
import yaml
from typing import Any, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

# Matches shell-style placeholders used in config.yaml:
#   ${VAR}            -> value of VAR, or empty string if unset
#   ${VAR:-default}   -> value of VAR, or "default" if unset
_ENV_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env_placeholders(value: Any) -> Any:
    """Recursively expand ``${VAR}`` / ``${VAR:-default}`` placeholders.

    Walks dicts, lists, and strings loaded from YAML and substitutes
    environment variables. This is the single mechanism by which config
    defaults flow: ``config.yaml`` carries the shell-style syntax and the
    values are resolved here at load time, so anything read via
    ``Config.get(...)`` sees the expanded value rather than the literal
    ``${VAR}`` placeholder string.

    Args:
        value: A value from the parsed YAML tree (dict, list, str, or scalar).

    Returns:
        The same structure with all string placeholders expanded.
    """
    if isinstance(value, dict):
        return {k: _expand_env_placeholders(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_placeholders(item) for item in value]
    if isinstance(value, str):
        def _replace(match: "re.Match[str]") -> str:
            var_name, default = match.group(1), match.group(2)
            return os.getenv(var_name, default if default is not None else "")

        return _ENV_PLACEHOLDER.sub(_replace, value)
    return value


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
        """Load configuration from the YAML file, expanding env placeholders.

        Defaults are expressed once, in ``config.yaml``, using the shell-style
        ``${VAR:-default}`` syntax. They are resolved against the environment
        here at load time — there is no second Python-side override mechanism.
        """
        config: Dict[str, Any] = {}

        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f) or {}

        return _expand_env_placeholders(config)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.

        Args:
            key: Configuration key (dot notation supported)
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split(".")
        value: Any = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_supabase_config(self) -> Dict[str, str]:
        """Get Supabase configuration."""
        return self.get("data.supabase", {})

    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration."""
        return self.get("api", {})

    def get_dashboard_config(self) -> Dict[str, Any]:
        """Get dashboard configuration."""
        return self.get("dashboard", {})

    def get_mlflow_config(self) -> Dict[str, str]:
        """Get MLflow configuration."""
        return self.get("mlflow", {})

    def get_paths(self) -> Dict[str, str]:
        """Get paths configuration."""
        return self.get("paths", {})

    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.get("models", {})

    def get_feature_config(self) -> Dict[str, Any]:
        """Get feature engineering configuration."""
        return self.get("features", {})


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get global configuration instance."""
    return config
