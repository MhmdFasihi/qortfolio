"""
Bitcoin Options Analytics Platform

A professional-grade platform for Bitcoin options analysis with Taylor expansion PnL simulation,
real-time data integration, and comprehensive risk management tools.

Author: Mhmd Fasihi
License: MIT
Python: 3.9+
"""

import sys
import logging
import warnings
from typing import Dict, Any, Optional

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

# Configure logging for the package
logger = logging.getLogger(__name__)

def check_dependencies() -> Dict[str, bool]:
    """
    Check if all required dependencies are available.
    
    Returns:
        Dict mapping dependency names to availability status
    """
    dependencies = {}
    
    # Core dependencies (must have)
    core_deps = [
        'pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn', 'plotly',
        'requests', 'aiohttp', 'streamlit', 'pyyaml', 'rich', 'numba', 'pydantic'
    ]
    
    # Optional dependencies for advanced features
    optional_deps = [
        ('schedule', 'Continuous data collection'),
        ('py_vollib', 'Professional options pricing'),
        ('QuantLib', 'Advanced quantitative finance'),
        ('ccxt', 'Multi-exchange connectivity'),
        ('redis', 'Caching'),
        ('sqlalchemy', 'Database operations'),
        ('python-dotenv', 'Environment variable management'),
        ('click', 'CLI interface'),
    ]
    
    # Check core dependencies
    for dep in core_deps:
        try:
            if dep == 'python-dotenv':
                import dotenv
            else:
                # Handle special import names
                import_name = dep.replace('-', '_')
                __import__(import_name)
            dependencies[dep] = True
        except ImportError:
            dependencies[dep] = False
            if dep in ['pandas', 'numpy', 'scipy']:  # Critical dependencies
                warnings.warn(f"Critical dependency '{dep}' not found. Install with: pip install {dep}")
    
    # Check optional dependencies
    for dep_info in optional_deps:
        if isinstance(dep_info, tuple):
            dep, description = dep_info
        else:
            dep, description = dep_info, ""
            
        try:
            if dep == 'python-dotenv':
                import dotenv
            elif dep == 'py_vollib':
                import py_vollib
            elif dep == 'QuantLib':
                import QuantLib
            else:
                import_name = dep.replace('-', '_')
                __import__(import_name)
            dependencies[dep] = True
        except ImportError:
            dependencies[dep] = False
            # Don't warn for optional dependencies
    
    return dependencies

def get_package_info() -> Dict[str, Any]:
    """
    Get comprehensive package information.
    
    Returns:
        Dict containing package metadata and dependency status
    """
    deps = check_dependencies()
    
    # Count available vs missing dependencies
    core_deps = ['pandas', 'numpy', 'scipy', 'matplotlib', 'requests']
    core_available = sum(1 for dep in core_deps if deps.get(dep, False))
    
    optional_available = sum(1 for dep in ['schedule', 'py_vollib', 'QuantLib'] if deps.get(dep, False))
    
    return {
        "name": "bitcoin-options-analytics",
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "license": __license__,
        "description": __description__,
        "python_requires": ">=3.9",
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "homepage": "https://github.com/MhmdFasihi/BTC-Option_Deribit",
        "dependencies": deps,
        "core_deps_available": f"{core_available}/{len(core_deps)}",
        "optional_deps_available": f"{optional_available}/3",
        "core_deps_ready": core_available >= 4,  # At least 4 out of 5 core deps
        "continuous_collector_ready": deps.get('schedule', False),
        "advanced_finance_ready": deps.get('py_vollib', False) and deps.get('QuantLib', False),
        "all_systems_ready": core_available >= 4 and deps.get('schedule', False),
    }

def configure_logging(level: str = "INFO", 
                     format_type: str = "standard",
                     filename: Optional[str] = None) -> None:
    """
    Configure package-wide logging.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Format type ('standard', 'detailed', 'json')
        filename: Optional log file path
    """
    # Define formats
    formats = {
        "standard": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "detailed": "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
        "json": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"  # Can be enhanced for JSON
    }
    
    log_format = formats.get(format_type, formats["standard"])
    
    # Configure basic logging
    logging_config = {
        'level': getattr(logging, level.upper()),
        'format': log_format,
        'datefmt': "%Y-%m-%d %H:%M:%S"
    }
    
    if filename:
        logging_config['filename'] = filename
        logging_config['filemode'] = 'a'  # Append mode
    
    logging.basicConfig(**logging_config)
    
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
        logger.info("✅ Continuous data collection available")
    else:
        logger.info("ℹ️  Continuous data collection requires 'schedule' package")
    
    if deps.get('py_vollib', False) and deps.get('QuantLib', False):
        logger.info("✅ Advanced financial models available")
    else:
        logger.info("ℹ️  Advanced financial models require 'py_vollib' and 'QuantLib'")

def print_startup_info():
    """Print startup information and status."""
    info = get_package_info()
    
    print(f"\n🚀 {info['description']}")
    print(f"📦 Version: {info['version']}")
    print(f"🐍 Python: {info['python_version']}")
    print(f"📊 Core Dependencies: {info['core_deps_available']}")
    print(f"⚙️  Optional Dependencies: {info['optional_deps_available']}")
    
    if info['core_deps_ready']:
        print("✅ System ready for basic operations")
    else:
        print("⚠️  Some core dependencies missing")
    
    if info['all_systems_ready']:
        print("🎯 All systems ready for production use")
    
    missing_critical = [dep for dep, available in info['dependencies'].items() 
                       if not available and dep in ['pandas', 'numpy', 'scipy']]
    
    if missing_critical:
        print(f"\n❌ Critical dependencies missing: {missing_critical}")
        print("💡 Install with: pip install -r requirements.txt")

# Version compatibility for setup.py
VERSION = __version__

# Initialize logging with default settings
try:
    configure_logging()
except Exception as e:
    # Fallback if logging configuration fails
    print(f"Warning: Logging configuration failed: {e}")
    logging.basicConfig(level=logging.INFO)

# Check dependencies on import (but don't fail)
try:
    _deps = check_dependencies()
    _missing_critical = [dep for dep in ['pandas', 'numpy', 'scipy'] if not _deps.get(dep, False)]
    
    if _missing_critical and __name__ != "__main__":
        print("⚠️  Bitcoin Options Analytics Platform")
        print(f"   Missing critical dependencies: {_missing_critical}")
        print("   Install with: pip install -r requirements.txt")
except Exception:
    # Don't fail on import errors during dependency checking
    pass

# For direct execution
if __name__ == "__main__":
    print("🧪 Bitcoin Options Analytics Platform - Package Info")
    print("=" * 60)
    print_startup_info()
    
    info = get_package_info()
    print("\n📋 Detailed Dependency Status:")
    print("-" * 40)
    for dep, available in info['dependencies'].items():
        status = "✅ Available" if available else "❌ Missing"
        print(f"  {dep:<20}: {status}")