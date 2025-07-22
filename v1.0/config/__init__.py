"""
# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com
Configuration management package for Bitcoin Options Analytics Platform.

This package handles all configuration-related functionality including:
- Environment variable management
- Settings validation
- Configuration file loading
- Logging configuration
"""

__all__ = [
    "Settings",
    "get_settings",
    "configure_logging",
]

# Package version
__version__ = "1.0.0"

# Re-export main configuration class when available
try:
    from .settings import Settings, get_settings
except ImportError:
    # Graceful fallback during initial setup
    Settings = None
    get_settings = None

try:
    from .logging_config import configure_logging
except ImportError:
    # Basic logging fallback
    import logging
    
    def configure_logging(level: str = "INFO") -> None:
        """Basic logging configuration fallback."""
        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )