"""
Bitcoin Options Analytics Platform

A professional-grade platform for Bitcoin options analysis with Taylor expansion PnL simulation,
real-time data integration, and comprehensive risk management tools.

Author: Options Analytics Team
License: MIT
Python: 3.9+
"""

from typing import Dict, Any
import logging

# Package metadata
__version__ = "1.0.0"
__author__ = "Mhmd Fasihi"
__email__ = "mhmd.fasihi1@gmail.com"
__license__ = "MIT"
__description__ = "Professional Bitcoin Options Analytics Platform with Taylor Expansion PnL Analysis"

# Package information
__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "__description__",
    "get_package_info",
    "configure_logging",
]

def get_package_info() -> Dict[str, Any]:
    """
    Get comprehensive package information.
    
    Returns:
        Dict containing package metadata
    """
    return {
        "name": "bitcoin-options-analytics",
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "license": __license__,
        "description": __description__,
        "python_requires": ">=3.9",
        "homepage": "https://github.com/MhmdFasihi/BTC-Option_Deribit",
    }

def configure_logging(level: str = "INFO") -> None:
    """
    Configure package-wide logging.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Create package logger
    logger = logging.getLogger(__name__)
    logger.info(f"Bitcoin Options Analytics Platform v{__version__} initialized")

# Package version for setup.py compatibility
VERSION = __version__

# Initialize logging with default level
configure_logging()