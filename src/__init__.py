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
import warnings

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
    "check_dependencies",
]

def check_dependencies() -> Dict[str, bool]:
    """
    Check if all required dependencies are available.
    
    Returns:
        Dict mapping dependency names to availability status
    """
    dependencies = {}
    
    # Core dependencies
    core_deps = [
        'pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn', 'plotly',
        'requests', 'aiohttp', 'streamlit', 'pyyaml', 'python-dotenv',
        'click', 'rich', 'numba', 'pydantic'
    ]
    
    # Optional dependencies for advanced features
    optional_deps = [
        'schedule',      # For continuous data collection
        'py_vollib',     # For professional options pricing
        'QuantLib',      # For advanced quantitative finance
        'ccxt',          # For multi-exchange connectivity
        'redis',         # For caching
        'sqlalchemy',    # For database operations
    ]
    
    # Check core dependencies
    for dep in core_deps:
        try:
            __import__(dep.replace('-', '_'))
            dependencies[dep] = True
        except ImportError:
            dependencies[dep] = False
            warnings.warn(f"Core dependency '{dep}' not found. Install with: pip install {dep}")
    
    # Check optional dependencies
    for dep in optional_deps:
        try:
            # Handle special cases
            if dep == 'python-dotenv':
                __import__('dotenv')
            elif dep == 'py_vollib':
                __import__('py_vollib')
            elif dep == 'QuantLib':
                __import__('QuantLib')
            else:
                __import__(dep.replace('-', '_'))
            dependencies[dep] = True
        except ImportError:
            dependencies[dep] = False
            # Don't warn for optional dependencies, just note
    
    return dependencies

def get_package_info() -> Dict[str, Any]:
    """
    Get comprehensive package information.
    
    Returns:
        Dict containing package metadata and dependency status
    """
    deps = check_dependencies()
    
    return {
        "name": "bitcoin-options-analytics",
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "license": __license__,
        "description": __description__,
        "python_requires": ">=3.9",
        "homepage": "https://github.com/MhmdFasihi/BTC-Option_Deribit",
        "dependencies": deps,
        "core_deps_available": all(deps.get(dep, False) for dep in [
            'pandas', 'numpy', 'scipy', 'matplotlib', 'requests'
        ]),
        "continuous_collector_ready": deps.get('schedule', False),
        "advanced_finance_ready": deps.get('py_vollib', False) and deps.get('QuantLib', False),
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
    
    # Log dependency status
    deps = check_dependencies()
    missing_core = [dep for dep in ['pandas', 'numpy', 'scipy', 'matplotlib', 'requests'] 
                    if not deps.get(dep, False)]
    
    if missing_core:
        logger.warning(f"Missing core dependencies: {missing_core}")
    else:
        logger.info("All core dependencies available")
    
    # Log optional feature availability
    if deps.get('schedule', False):
        logger.info("Continuous data collection available")
    else:
        logger.info("Continuous data collection requires 'schedule' package")
    
    if deps.get('py_vollib', False) and deps.get('QuantLib', False):
        logger.info("Advanced financial models available")
    else:
        logger.info("Advanced financial models require 'py_vollib' and 'QuantLib'")

# Package version for setup.py compatibility
VERSION = __version__

# Initialize logging with default level and dependency check
configure_logging()

# Print a friendly startup message if dependencies are missing
_deps = check_dependencies()
_missing_critical = [dep for dep in ['pandas', 'numpy', 'scipy'] if not _deps.get(dep, False)]

if _missing_critical:
    print("⚠️  Missing critical dependencies. Install with:")
    print(f"   pip install {' '.join(_missing_critical)}")
    print("   Or: pip install -r requirements.txt")