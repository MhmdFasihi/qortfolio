#!/usr/bin/env python3
"""
Bitcoin Options Analytics Platform - Setup Configuration
Backward compatibility setup.py for systems that don't support pyproject.toml
"""

from setuptools import setup, find_packages
import os
import sys

# Ensure we're using Python 3.9+
if sys.version_info < (3, 9):
    raise RuntimeError("This package requires Python 3.9 or later")

# Read the contents of README file
current_dir = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(current_dir, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Read requirements from requirements.txt
def read_requirements(filename):
    """Read requirements from a file."""
    requirements_path = os.path.join(current_dir, filename)
    try:
        with open(requirements_path, encoding="utf-8") as f:
            requirements = []
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if line and not line.startswith("#") and not line.startswith("-r"):
                    # Remove version specifiers comments
                    if " #" in line:
                        line = line.split(" #")[0].strip()
                    requirements.append(line)
            return requirements
    except FileNotFoundError:
        print(f"Warning: {filename} not found")
        return []

# Read version from src/__init__.py
def get_version():
    """Extract version from src/__init__.py."""
    version_file = os.path.join(current_dir, "src", "__init__.py")
    try:
        with open(version_file, encoding="utf-8") as f:
            for line in f:
                if line.startswith("__version__"):
                    return line.split("=")[1].strip().strip("\"'")
    except FileNotFoundError:
        pass
    return "1.0.0"

# Core dependencies
install_requires = [
    "pandas>=2.0.0,<3.0.0",
    "numpy>=1.24.0,<2.0.0",
    "scipy>=1.10.0,<2.0.0",
    "py_vollib>=1.0.1",
    "QuantLib-Python>=1.17,<1.32",
    "mibian>=0.1.3",
    "requests>=2.31.0,<3.0.0",
    "aiohttp>=3.8.0,<4.0.0",
    "matplotlib>=3.7.0,<4.0.0",
    "seaborn>=0.12.0,<1.0.0",
    "plotly>=5.17.0,<6.0.0",
    "streamlit>=1.28.0,<2.0.0",
    "pyyaml>=6.0,<7.0",
    "python-dotenv>=1.0.0,<2.0.0",
    "click>=8.1.0,<9.0.0",
    "rich>=13.4.0,<14.0.0",
    "numba>=0.57.0,<1.0.0",
    "pydantic>=2.4.0,<3.0.0",
    "python-dateutil>=2.8.0,<3.0.0",
    "structlog>=23.1.0"
]

# Optional dependencies
extras_require = {
    "dev": [
        "pytest>=7.4.0,<8.0.0",
        "pytest-cov>=4.1.0,<5.0.0",
        "pytest-asyncio>=0.21.0,<1.0.0",
        "black>=23.7.0,<24.0.0",
        "isort>=5.12.0,<6.0.0",
        "flake8>=6.0.0,<7.0.0",
        "mypy>=1.5.0,<2.0.0",
        "pre-commit>=3.3.0,<4.0.0",
        "jupyter>=1.0.0,<2.0.0",
        "jupyterlab>=4.0.0,<5.0.0"
    ],
    "ml": [
        "scikit-learn>=1.3.0,<2.0.0",
        "xgboost>=1.7.0,<2.0.0",
        "tensorflow>=2.13.0,<3.0.0",
        "torch>=2.0.0"
    ],
    "database": [
        "sqlalchemy>=2.0.0,<3.0.0",
        "redis>=4.6.0,<5.0.0",
        "psycopg2-binary>=2.9.0,<3.0.0",
        "alembic>=1.11.0,<2.0.0"
    ],
    "crypto": [
        "ccxt>=4.0.0",
        "python-binance>=1.0.19",
        "websocket-client>=1.6.0",
        "cryptofeed>=2.4.0"
    ],
    "docs": [
        "sphinx>=7.1.0,<8.0.0",
        "sphinx-rtd-theme>=1.3.0,<2.0.0",
        "sphinx-autoapi>=2.1.0,<3.0.0",
        "myst-parser>=2.0.0,<3.0.0"
    ]
}

# Add 'all' extra that includes everything
extras_require["all"] = list(set(
    req for extra_deps in extras_require.values() 
    for req in extra_deps
))

setup(
    # Basic package information
    name="bitcoin-options-analytics",
    version=get_version(),
    author="Mhmd Fasihi",
    author_email="mhmd.fasihi1@gmail.com",
    description="Professional Bitcoin Options Analytics Platform with Taylor Expansion PnL Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MhmdFasihi/BTC-Option_Deribit",
    project_urls={
        "Documentation": "https://github.com/MhmdFasihi/BTC-Option_Deribit/wiki",
        "Source": "https://github.com/MhmdFasihi/BTC-Option_Deribit",
        "Tracker": "https://github.com/MhmdFasihi/BTC-Option_Deribit/issues",
    },
    
    # Package configuration
    packages=find_packages(),
    package_dir={"": "."},
    package_data={
        "": ["*.yml", "*.yaml", "*.json", "*.toml"],
        "config": ["*.yml", "*.yaml", "*.json"],
        "src": ["py.typed"],  # PEP 561 type information
    },
    include_package_data=True,
    
    # Dependencies
    install_requires=install_requires,
    extras_require=extras_require,
    python_requires=">=3.9",
    
    # Entry points
    entry_points={
        "console_scripts": [
            "btc-options=src.cli:main",
            "options-analyzer=src.cli:analyzer_main",
            "pnl-simulator=src.cli:pnl_main",
        ],
        "gui_scripts": [
            "options-dashboard=dashboard.app:main",
        ],
    },
    
    # Classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business :: Financial",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Typing :: Typed",
    ],
    
    # Additional metadata
    license="MIT",
    keywords=[
        "bitcoin", "options", "derivatives", "quantitative-finance",
        "black-scholes", "greeks", "pnl-analysis", "taylor-expansion",
        "cryptocurrency", "trading"
    ],
    
    # Development status
    zip_safe=False,  # Required for some packages
    
    # Test suite
    test_suite="tests",
    tests_require=extras_require["dev"],
)

# Post-installation messages
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Bitcoin Options Analytics Platform")
    print("="*60)
    print("✅ Installation completed successfully!")
    print("\n📋 Next steps:")
    print("   1. Copy .env.example to .env and configure your settings")
    print("   2. Set up your API credentials in the .env file")
    print("   3. Run: python -m pytest tests/ to verify installation")
    print("   4. Start dashboard: streamlit run dashboard/app.py")
    print("\n📚 Documentation: https://bitcoin-options-analytics.readthedocs.io/")
    print("🐛 Issues: https://github.com/yourusername/bitcoin-options-analytics/issues")
    print("="*60)