"""Core fraud detection utilities and configuration management."""

import os
import random
import logging
from typing import Any, Dict, Optional, Union
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device for computation.
    
    Returns:
        torch.device: CUDA, MPS, or CPU device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Setup logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    # Set timezone for consistent timestamps
    os.environ["TZ"] = "UTC"
    
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=handlers,
        datefmt="%Y-%m-%d %H:%M:%S UTC"
    )


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dict containing configuration parameters
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    return OmegaConf.load(config_path)


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    OmegaConf.save(config, config_path)


def hash_sensitive_data(data: str, salt: str = "fraud_detection_salt") -> str:
    """Hash sensitive data for privacy protection.
    
    Args:
        data: Sensitive string to hash
        salt: Salt for hashing
        
    Returns:
        Hashed string
    """
    import hashlib
    
    return hashlib.sha256(f"{data}{salt}".encode()).hexdigest()[:16]


def obfuscate_ip(ip: str) -> str:
    """Obfuscate IP address for privacy.
    
    Args:
        ip: IP address to obfuscate
        
    Returns:
        Obfuscated IP address
    """
    parts = ip.split(".")
    if len(parts) == 4:
        # Keep first two octets, obfuscate last two
        return f"{parts[0]}.{parts[1]}.xxx.xxx"
    return "xxx.xxx.xxx.xxx"


def obfuscate_email(email: str) -> str:
    """Obfuscate email address for privacy.
    
    Args:
        email: Email address to obfuscate
        
    Returns:
        Obfuscated email address
    """
    if "@" in email:
        local, domain = email.split("@", 1)
        if len(local) > 2:
            return f"{local[:2]}***@{domain}"
        else:
            return f"***@{domain}"
    return "***@***.com"


class Config:
    """Configuration management class."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self._config = {}
        
        if config_path:
            self.load(config_path)
    
    def load(self, config_path: Union[str, Path]) -> None:
        """Load configuration from file.
        
        Args:
            config_path: Path to configuration file
        """
        self._config = load_config(config_path)
        self.config_path = config_path
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split(".")
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """Save configuration to file.
        
        Args:
            config_path: Optional path to save configuration
        """
        path = config_path or self.config_path
        if path:
            save_config(self._config, path)
