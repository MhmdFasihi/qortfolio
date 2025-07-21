#!/usr/bin/env python3
"""
# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com
Asset Configuration Management

This module handles loading and managing cryptocurrency asset configurations
discovered from Deribit exchange.

Features:
- Load asset configurations from JSON files
- Provide default fallbacks if config is missing
- Validate asset availability
- Cache configurations for performance
- Auto-refresh capabilities
"""

import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
import threading

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class AssetInfo:
    """Information about a tradeable asset."""
    symbol: str
    supports_options: bool
    current_price: Optional[float]
    instruments_count: int
    last_updated: str
    is_recommended: bool = False
    
    def __post_init__(self):
        """Validate asset info after initialization."""
        if not self.symbol:
            raise ValueError("Asset symbol cannot be empty")
        if self.instruments_count < 0:
            raise ValueError("Instruments count cannot be negative")

class AssetsConfigError(Exception):
    """Exception raised for asset configuration errors."""
    pass

class AssetsConfig:
    """
    Manages cryptocurrency asset configurations for trading.
    
    This class loads asset configurations from discovery files and provides
    a clean interface for getting available trading currencies.
    """
    
    def __init__(self, 
                 config_file: str = None,
                 auto_refresh_hours: int = 24,
                 fallback_currencies: List[str] = None):
        """
        Initialize asset configuration manager.
        
        Args:
            config_file: Path to asset configuration file (auto-detected if None)
            auto_refresh_hours: Hours after which to consider config stale
            fallback_currencies: Currencies to use if config is unavailable
        """
        # Smart config file path detection
        if config_file is None:
            config_file = self._find_config_file()
        
        self.config_file = Path(config_file)
        self.auto_refresh_hours = auto_refresh_hours
        self.fallback_currencies = fallback_currencies or ['BTC', 'ETH']
        
        # Internal state
        self._config_data: Optional[Dict[str, Any]] = None
        self._last_loaded: Optional[datetime] = None
        self._lock = threading.RLock()
        self._fallback_mode = False
        self._config_load_attempted = False
        
        # Asset information cache
        self._assets_cache: Dict[str, AssetInfo] = {}
        
        logger.debug(f"AssetConfig initialized with config file: {self.config_file}")

    def _find_config_file(self) -> str:
        """
        Intelligently find the configuration file in various possible locations.
        
        Returns:
            Path to configuration file
        """
        # Possible locations to check (in order of preference)
        possible_paths = [
            "config/assets_config.json",           # Standard location
            "assets_config.json",                  # Root directory
            "../config/assets_config.json",       # If running from subdirectory
            "src/config/assets_config.json",      # Alternative structure
            os.path.expanduser("~/.btc-options/assets_config.json"),  # User home
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                logger.debug(f"Found config file at: {path}")
                return path
        
        # If none found, use the standard location (will be created later)
        default_path = "config/assets_config.json"
        logger.debug(f"No existing config found, will use: {default_path}")
        return default_path

    def _load_config_file(self) -> Dict[str, Any]:
        """
        Load configuration from JSON file.
        
        Returns:
            Configuration dictionary
            
        Raises:
            AssetsConfigError: If file cannot be loaded or is invalid
        """
        if not self.config_file.exists():
            raise AssetsConfigError(f"Configuration file not found: {self.config_file}")
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Validate config structure
            if not isinstance(config, dict):
                raise AssetsConfigError("Config file must contain a JSON object")
            
            # Check for required top-level keys with minimal validation
            if 'assets' not in config:
                config['assets'] = {}
            
            if 'metadata' not in config:
                config['metadata'] = {
                    'last_updated': datetime.now().isoformat(),
                    'discovery_script_version': 'unknown'
                }
            
            # Ensure assets section has required keys
            assets = config['assets']
            for key in ['options_enabled', 'default_currencies']:
                if key not in assets:
                    if key == 'options_enabled':
                        assets[key] = self.fallback_currencies.copy()
                    elif key == 'default_currencies':
                        assets[key] = self.fallback_currencies.copy()
            
            logger.debug(f"Loaded configuration with {len(assets.get('options_enabled', []))} options-enabled currencies")
            return config
            
        except json.JSONDecodeError as e:
            raise AssetsConfigError(f"Invalid JSON in config file: {e}")
        except Exception as e:
            raise AssetsConfigError(f"Failed to load config file: {e}")

    def _is_config_stale(self) -> bool:
        """
        Check if the current configuration is stale and needs refreshing.
        
        Returns:
            True if config should be refreshed
        """
        if not self._last_loaded or not self._config_data:
            return True
        
        # Check if file was modified since last load
        try:
            if self.config_file.exists():
                file_mtime = datetime.fromtimestamp(self.config_file.stat().st_mtime)
                if file_mtime > self._last_loaded:
                    return True
        except Exception:
            pass
        
        # Check if auto-refresh period has elapsed
        if (datetime.now() - self._last_loaded).total_seconds() > (self.auto_refresh_hours * 3600):
            return True
        
        return False

    def _load_config(self, force_reload: bool = False) -> Dict[str, Any]:
        """
        Load or reload configuration with caching and reduced logging.
        
        Args:
            force_reload: Force reload even if cache is valid
            
        Returns:
            Configuration dictionary
        """
        with self._lock:
            # If we're in fallback mode and not forced, return cached fallback
            if self._fallback_mode and not force_reload and self._config_data:
                return self._config_data
            
            # If config is fresh and we're not forced, return cached
            if not force_reload and not self._is_config_stale() and self._config_data:
                return self._config_data
            
            # Only attempt to load if we haven't tried yet or forced
            if not self._config_load_attempted or force_reload:
                self._config_load_attempted = True
                
                try:
                    logger.debug("Loading asset configuration...")
                    self._config_data = self._load_config_file()
                    self._last_loaded = datetime.now()
                    self._fallback_mode = False
                    
                    # Update assets cache
                    self._update_assets_cache()
                    
                    logger.info("Asset configuration loaded successfully")
                    return self._config_data
                    
                except AssetsConfigError:
                    # Only log once when first entering fallback mode
                    if not self._fallback_mode:
                        logger.info("Config file not found, using fallback configuration with BTC, ETH")
                        self._fallback_mode = True
                    
                    # Create and cache fallback config
                    self._config_data = self._create_fallback_config()
                    self._last_loaded = datetime.now()
                    self._update_assets_cache()
                    
            return self._config_data or self._create_fallback_config()

    def _create_fallback_config(self) -> Dict[str, Any]:
        """
        Create a fallback configuration when the main config is unavailable.
        
        Returns:
            Fallback configuration dictionary
        """
        return {
            'assets': {
                'all_currencies': self.fallback_currencies.copy(),
                'options_enabled': self.fallback_currencies.copy(),
                'recommended': self.fallback_currencies.copy(),
                'default_currencies': self.fallback_currencies.copy()
            },
            'prices': {currency: None for currency in self.fallback_currencies},
            'details': {
                currency: {
                    'currency': currency,
                    'supports_options': True,
                    'instruments_count': 1,  # Assume at least some options
                    'is_fallback': True
                }
                for currency in self.fallback_currencies
            },
            'metadata': {
                'last_updated': datetime.now().isoformat(),
                'is_fallback': True,
                'discovery_script_version': 'fallback'
            }
        }

    def _update_assets_cache(self) -> None:
        """Update the internal assets cache from loaded configuration."""
        if not self._config_data:
            return
        
        self._assets_cache.clear()
        
        assets = self._config_data.get('assets', {})
        prices = self._config_data.get('prices', {})
        details = self._config_data.get('details', {})
        recommended = set(assets.get('recommended', []))
        
        for currency in assets.get('options_enabled', []):
            currency_details = details.get(currency, {})
            
            asset_info = AssetInfo(
                symbol=currency,
                supports_options=currency_details.get('supports_options', True),
                current_price=prices.get(currency),
                instruments_count=currency_details.get('instruments_count', 0),
                last_updated=self._config_data.get('metadata', {}).get('last_updated', ''),
                is_recommended=currency in recommended
            )
            
            self._assets_cache[currency] = asset_info

    def get_all_currencies(self) -> List[str]:
        """Get all available currencies."""
        try:
            config = self._load_config()
            return config.get('assets', {}).get('all_currencies', self.fallback_currencies.copy())
        except Exception:
            return self.fallback_currencies.copy()

    def get_options_enabled_currencies(self) -> List[str]:
        """Get currencies that support options trading."""
        try:
            config = self._load_config()
            return config.get('assets', {}).get('options_enabled', self.fallback_currencies.copy())
        except Exception:
            return self.fallback_currencies.copy()

    def get_recommended_currencies(self) -> List[str]:
        """Get currencies recommended for trading based on liquidity and activity."""
        try:
            config = self._load_config()
            recommended = config.get('assets', {}).get('recommended', [])
            
            if not recommended:
                recommended = self.get_options_enabled_currencies()
            
            return recommended
        except Exception:
            return self.fallback_currencies.copy()

    def get_default_currencies(self) -> List[str]:
        """Get default currencies for trading (usually the top recommended ones)."""
        try:
            config = self._load_config()
            defaults = config.get('assets', {}).get('default_currencies', [])
            
            if not defaults:
                recommended = self.get_recommended_currencies()
                defaults = recommended[:3] if len(recommended) >= 3 else recommended
                if not defaults:
                    defaults = self.fallback_currencies.copy()
            
            return defaults
        except Exception:
            return self.fallback_currencies.copy()

    def get_asset_info(self, currency: str) -> Optional[AssetInfo]:
        """Get detailed information about a specific asset."""
        try:
            self._load_config()  # Ensure cache is up to date
            return self._assets_cache.get(currency.upper())
        except Exception:
            return None

    def get_current_price(self, currency: str) -> Optional[float]:
        """Get current price for a currency."""
        try:
            config = self._load_config()
            prices = config.get('prices', {})
            return prices.get(currency.upper())
        except Exception:
            return None

    def get_currencies_with_minimum_options(self, min_options: int = 10) -> List[str]:
        """Get currencies that have at least the specified number of options."""
        try:
            self._load_config()  # Ensure cache is up to date
            
            result = []
            for currency, info in self._assets_cache.items():
                if info.instruments_count >= min_options:
                    result.append(currency)
            
            return sorted(result)
        except Exception:
            return []

    def is_currency_supported(self, currency: str) -> bool:
        """Check if a currency is supported for options trading."""
        try:
            options_enabled = self.get_options_enabled_currencies()
            return currency.upper() in [c.upper() for c in options_enabled]
        except Exception:
            return currency.upper() in [c.upper() for c in self.fallback_currencies]

    def validate_currencies(self, currencies: List[str]) -> Dict[str, bool]:
        """Validate a list of currencies against available options."""
        result = {}
        try:
            options_enabled = [c.upper() for c in self.get_options_enabled_currencies()]
            
            for currency in currencies:
                result[currency] = currency.upper() in options_enabled
                
        except Exception:
            # Fallback validation
            fallback_upper = [c.upper() for c in self.fallback_currencies]
            for currency in currencies:
                result[currency] = currency.upper() in fallback_upper
        
        return result

    def get_config_info(self) -> Dict[str, Any]:
        """Get information about the current configuration."""
        try:
            config = self._load_config()
            metadata = config.get('metadata', {})
            
            return {
                'config_file': str(self.config_file),
                'config_exists': self.config_file.exists(),
                'last_loaded': self._last_loaded.isoformat() if self._last_loaded else None,
                'is_fallback': metadata.get('is_fallback', False),
                'last_discovery': metadata.get('last_updated', 'Unknown'),
                'discovery_version': metadata.get('discovery_script_version', 'Unknown'),
                'total_currencies': len(self.get_all_currencies()),
                'options_enabled_count': len(self.get_options_enabled_currencies()),
                'recommended_count': len(self.get_recommended_currencies()),
                'auto_refresh_hours': self.auto_refresh_hours,
                'fallback_mode': self._fallback_mode
            }
        except Exception as e:
            return {
                'config_file': str(self.config_file),
                'config_exists': False,
                'error': str(e),
                'is_fallback': True,
                'fallback_mode': True
            }

    def refresh_config(self) -> bool:
        """Force refresh the configuration from file."""
        try:
            self._config_load_attempted = False  # Allow retry
            self._load_config(force_reload=True)
            logger.info("Configuration refreshed successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to refresh configuration: {e}")
            return False

    def print_status(self) -> None:
        """Print current configuration status to console."""
        try:
            print("\n📊 ASSET CONFIGURATION STATUS")
            print("=" * 50)
            
            config_info = self.get_config_info()
            
            print(f"Config File: {config_info['config_file']}")
            print(f"File Exists: {'✅ Yes' if config_info['config_exists'] else '❌ No'}")
            print(f"Last Loaded: {config_info.get('last_loaded', 'Never')}")
            
            if config_info.get('fallback_mode', False):
                print(f"Mode: ⚠️  Fallback Mode (using BTC, ETH defaults)")
                print(f"💡 Run 'python discover_deribit_assets.py' to discover available currencies")
            else:
                print(f"Mode: ✅ Configuration Loaded")
                print(f"Last Discovery: {config_info.get('last_discovery', 'Unknown')}")
            
            print(f"\n📈 AVAILABLE CURRENCIES:")
            print(f"Total: {config_info.get('total_currencies', 0)}")
            print(f"Options Enabled: {config_info.get('options_enabled_count', 0)}")
            print(f"Recommended: {config_info.get('recommended_count', 0)}")
            
            defaults = self.get_default_currencies()
            recommended = self.get_recommended_currencies()
            
            print(f"\n🎯 DEFAULT CURRENCIES: {', '.join(defaults)}")
            print(f"🌟 RECOMMENDED: {', '.join(recommended)}")
            
            if not config_info.get('fallback_mode', False):
                print("\n📋 CURRENCY DETAILS:")
                print("-" * 30)
                for currency in defaults:
                    info = self.get_asset_info(currency)
                    if info:
                        price_str = f"${info.current_price:,.2f}" if info.current_price else "N/A"
                        print(f"{currency}: {info.instruments_count} options, price: {price_str}")
                    else:
                        print(f"{currency}: Standard fallback currency")
                        
        except Exception as e:
            print(f"❌ Error printing status: {e}")


# Global instance management with thread safety
_global_assets_config: Optional[AssetsConfig] = None
_global_config_lock = threading.Lock()

def get_assets_config(config_file: str = None) -> AssetsConfig:
    """Get the global assets configuration instance (thread-safe)."""
    global _global_assets_config
    
    with _global_config_lock:
        if _global_assets_config is None:
            _global_assets_config = AssetsConfig(config_file)
        
        return _global_assets_config

def reset_global_config():
    """Reset the global configuration instance (for testing)."""
    global _global_assets_config
    with _global_config_lock:
        _global_assets_config = None

# Convenience functions for common operations (with minimal logging)
def get_default_currencies() -> List[str]:
    """Get default trading currencies."""
    try:
        return get_assets_config().get_default_currencies()
    except Exception:
        return ['BTC', 'ETH']  # Safe fallback

def get_recommended_currencies() -> List[str]:
    """Get recommended trading currencies."""
    try:
        return get_assets_config().get_recommended_currencies()
    except Exception:
        return ['BTC', 'ETH']  # Safe fallback

def get_options_enabled_currencies() -> List[str]:
    """Get all options-enabled currencies."""
    try:
        return get_assets_config().get_options_enabled_currencies()
    except Exception:
        return ['BTC', 'ETH']  # Safe fallback

def is_currency_supported(currency: str) -> bool:
    """Check if a currency supports options trading."""
    try:
        return get_assets_config().is_currency_supported(currency)
    except Exception:
        return currency.upper() in ['BTC', 'ETH']  # Safe fallback

def validate_currencies(currencies: List[str]) -> Dict[str, bool]:
    """Validate a list of currencies."""
    try:
        return get_assets_config().validate_currencies(currencies)
    except Exception:
        # Safe fallback
        return {currency: currency.upper() in ['BTC', 'ETH'] for currency in currencies}

if __name__ == "__main__":
    # Test the configuration system
    import sys
    
    print("🧪 Testing Asset Configuration System")
    print("=" * 50)
    
    try:
        config = AssetsConfig()
        config.print_status()
        
        print("\n✅ Asset configuration system is working!")
        
        # Test convenience functions
        print(f"\n🧪 Testing convenience functions:")
        print(f"Default currencies: {get_default_currencies()}")
        print(f"BTC supported: {is_currency_supported('BTC')}")
        print(f"ETH supported: {is_currency_supported('ETH')}")
        
    except Exception as e:
        print(f"❌ Error testing configuration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)