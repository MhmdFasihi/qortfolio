#!/usr/bin/env python3
"""
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
                 config_file: str = "config/assets_config.json",
                 auto_refresh_hours: int = 24,
                 fallback_currencies: List[str] = None):
        """
        Initialize asset configuration manager.
        
        Args:
            config_file: Path to asset configuration file
            auto_refresh_hours: Hours after which to consider config stale
            fallback_currencies: Currencies to use if config is unavailable
        """
        self.config_file = Path(config_file)
        self.auto_refresh_hours = auto_refresh_hours
        self.fallback_currencies = fallback_currencies or ['BTC', 'ETH']
        
        # Internal state
        self._config_data: Optional[Dict[str, Any]] = None
        self._last_loaded: Optional[datetime] = None
        self._lock = threading.RLock()
        
        # Asset information cache
        self._assets_cache: Dict[str, AssetInfo] = {}
        
        logger.info(f"AssetConfig initialized with config file: {self.config_file}")

    def _load_config_file(self) -> Dict[str, Any]:
        """
        Load configuration from JSON file.
        
        Returns:
            Configuration dictionary
            
        Raises:
            AssetsConfigError: If file cannot be loaded or is invalid
        """
        try:
            if not self.config_file.exists():
                raise AssetsConfigError(f"Configuration file not found: {self.config_file}")
            
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            
            # Validate config structure
            required_keys = ['assets', 'metadata']
            missing_keys = [key for key in required_keys if key not in config]
            if missing_keys:
                raise AssetsConfigError(f"Missing required config keys: {missing_keys}")
            
            # Validate assets section
            assets = config['assets']
            required_asset_keys = ['options_enabled', 'default_currencies']
            missing_asset_keys = [key for key in required_asset_keys if key not in assets]
            if missing_asset_keys:
                raise AssetsConfigError(f"Missing required asset keys: {missing_asset_keys}")
            
            logger.info(f"Loaded configuration with {len(assets.get('options_enabled', []))} options-enabled currencies")
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
        if not self._last_loaded:
            return True
        
        if not self._config_data:
            return True
        
        # Check if file was modified since last load
        try:
            file_mtime = datetime.fromtimestamp(self.config_file.stat().st_mtime)
            if file_mtime > self._last_loaded:
                logger.info("Config file was modified, marking as stale")
                return True
        except Exception:
            # If we can't check file time, assume it's stale
            return True
        
        # Check if auto-refresh period has elapsed
        if (datetime.now() - self._last_loaded).total_seconds() > (self.auto_refresh_hours * 3600):
            logger.info(f"Config is older than {self.auto_refresh_hours} hours, marking as stale")
            return True
        
        return False

    def _load_config(self, force_reload: bool = False) -> Dict[str, Any]:
        """
        Load or reload configuration with caching.
        
        Args:
            force_reload: Force reload even if cache is valid
            
        Returns:
            Configuration dictionary
        """
        with self._lock:
            if force_reload or self._is_config_stale():
                try:
                    logger.info("Loading asset configuration...")
                    self._config_data = self._load_config_file()
                    self._last_loaded = datetime.now()
                    
                    # Update assets cache
                    self._update_assets_cache()
                    
                    logger.info("Asset configuration loaded successfully")
                    
                except AssetsConfigError as e:
                    logger.error(f"Failed to load config: {e}")
                    
                    # If we have cached data, use it with a warning
                    if self._config_data:
                        logger.warning("Using cached configuration data")
                    else:
                        logger.warning("No cached data available, using fallback configuration")
                        self._config_data = self._create_fallback_config()
                        
            return self._config_data or self._create_fallback_config()

    def _create_fallback_config(self) -> Dict[str, Any]:
        """
        Create a fallback configuration when the main config is unavailable.
        
        Returns:
            Fallback configuration dictionary
        """
        logger.warning(f"Creating fallback configuration with currencies: {self.fallback_currencies}")
        
        return {
            'assets': {
                'all_currencies': self.fallback_currencies,
                'options_enabled': self.fallback_currencies,
                'recommended': self.fallback_currencies,
                'default_currencies': self.fallback_currencies
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
        
        logger.debug(f"Updated assets cache with {len(self._assets_cache)} currencies")

    def get_all_currencies(self) -> List[str]:
        """
        Get all available currencies.
        
        Returns:
            List of all currency symbols
        """
        config = self._load_config()
        return config.get('assets', {}).get('all_currencies', self.fallback_currencies)

    def get_options_enabled_currencies(self) -> List[str]:
        """
        Get currencies that support options trading.
        
        Returns:
            List of options-enabled currency symbols
        """
        config = self._load_config()
        return config.get('assets', {}).get('options_enabled', self.fallback_currencies)

    def get_recommended_currencies(self) -> List[str]:
        """
        Get currencies recommended for trading based on liquidity and activity.
        
        Returns:
            List of recommended currency symbols
        """
        config = self._load_config()
        recommended = config.get('assets', {}).get('recommended', [])
        
        # If no recommended currencies, fall back to options-enabled
        if not recommended:
            recommended = self.get_options_enabled_currencies()
        
        return recommended

    def get_default_currencies(self) -> List[str]:
        """
        Get default currencies for trading (usually the top recommended ones).
        
        Returns:
            List of default currency symbols
        """
        config = self._load_config()
        defaults = config.get('assets', {}).get('default_currencies', [])
        
        # If no defaults specified, use top 3 recommended or fallback
        if not defaults:
            recommended = self.get_recommended_currencies()
            defaults = recommended[:3] if len(recommended) >= 3 else recommended
            if not defaults:
                defaults = self.fallback_currencies
        
        return defaults

    def get_asset_info(self, currency: str) -> Optional[AssetInfo]:
        """
        Get detailed information about a specific asset.
        
        Args:
            currency: Currency symbol
            
        Returns:
            AssetInfo object or None if not found
        """
        self._load_config()  # Ensure cache is up to date
        return self._assets_cache.get(currency.upper())

    def get_current_price(self, currency: str) -> Optional[float]:
        """
        Get current price for a currency.
        
        Args:
            currency: Currency symbol
            
        Returns:
            Current price or None if not available
        """
        config = self._load_config()
        prices = config.get('prices', {})
        return prices.get(currency.upper())

    def get_currencies_with_minimum_options(self, min_options: int = 10) -> List[str]:
        """
        Get currencies that have at least the specified number of options.
        
        Args:
            min_options: Minimum number of options required
            
        Returns:
            List of currency symbols meeting the criteria
        """
        self._load_config()  # Ensure cache is up to date
        
        result = []
        for currency, info in self._assets_cache.items():
            if info.instruments_count >= min_options:
                result.append(currency)
        
        return sorted(result)

    def is_currency_supported(self, currency: str) -> bool:
        """
        Check if a currency is supported for options trading.
        
        Args:
            currency: Currency symbol to check
            
        Returns:
            True if currency supports options trading
        """
        options_enabled = self.get_options_enabled_currencies()
        return currency.upper() in [c.upper() for c in options_enabled]

    def validate_currencies(self, currencies: List[str]) -> Dict[str, bool]:
        """
        Validate a list of currencies against available options.
        
        Args:
            currencies: List of currency symbols to validate
            
        Returns:
            Dictionary mapping currency to validation result
        """
        result = {}
        options_enabled = [c.upper() for c in self.get_options_enabled_currencies()]
        
        for currency in currencies:
            result[currency] = currency.upper() in options_enabled
        
        return result

    def get_config_info(self) -> Dict[str, Any]:
        """
        Get information about the current configuration.
        
        Returns:
            Dictionary with configuration metadata
        """
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
            'auto_refresh_hours': self.auto_refresh_hours
        }

    def refresh_config(self) -> bool:
        """
        Force refresh the configuration from file.
        
        Returns:
            True if refresh was successful
        """
        try:
            self._load_config(force_reload=True)
            logger.info("Configuration refreshed successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to refresh configuration: {e}")
            return False

    def print_status(self) -> None:
        """Print current configuration status to console."""
        print("\n📊 ASSET CONFIGURATION STATUS")
        print("=" * 50)
        
        config_info = self.get_config_info()
        
        print(f"Config File: {config_info['config_file']}")
        print(f"File Exists: {'✅ Yes' if config_info['config_exists'] else '❌ No'}")
        print(f"Last Loaded: {config_info['last_loaded'] or 'Never'}")
        print(f"Is Fallback: {'⚠️  Yes' if config_info['is_fallback'] else '✅ No'}")
        print(f"Last Discovery: {config_info['last_discovery']}")
        
        print(f"\n📈 AVAILABLE CURRENCIES:")
        print(f"Total: {config_info['total_currencies']}")
        print(f"Options Enabled: {config_info['options_enabled_count']}")
        print(f"Recommended: {config_info['recommended_count']}")
        
        print(f"\n🎯 DEFAULT CURRENCIES: {', '.join(self.get_default_currencies())}")
        print(f"🌟 RECOMMENDED: {', '.join(self.get_recommended_currencies())}")
        
        print("\n📋 CURRENCY DETAILS:")
        print("-" * 30)
        for currency in self.get_default_currencies():
            info = self.get_asset_info(currency)
            if info:
                price_str = f"${info.current_price:,.2f}" if info.current_price else "N/A"
                print(f"{currency}: {info.instruments_count} options, price: {price_str}")


# Global instance for easy access
_global_assets_config: Optional[AssetsConfig] = None

def get_assets_config(config_file: str = "config/assets_config.json") -> AssetsConfig:
    """
    Get the global assets configuration instance.
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        AssetsConfig instance
    """
    global _global_assets_config
    
    if _global_assets_config is None:
        _global_assets_config = AssetsConfig(config_file)
    
    return _global_assets_config

# Convenience functions for common operations
def get_default_currencies() -> List[str]:
    """Get default trading currencies."""
    return get_assets_config().get_default_currencies()

def get_recommended_currencies() -> List[str]:
    """Get recommended trading currencies."""
    return get_assets_config().get_recommended_currencies()

def get_options_enabled_currencies() -> List[str]:
    """Get all options-enabled currencies."""
    return get_assets_config().get_options_enabled_currencies()

def is_currency_supported(currency: str) -> bool:
    """Check if a currency supports options trading."""
    return get_assets_config().is_currency_supported(currency)

def validate_currencies(currencies: List[str]) -> Dict[str, bool]:
    """Validate a list of currencies."""
    return get_assets_config().validate_currencies(currencies)

if __name__ == "__main__":
    # Test the configuration system
    import sys
    
    print("🧪 Testing Asset Configuration System")
    print("=" * 50)
    
    try:
        config = AssetsConfig()
        config.print_status()
        
        print("\n✅ Asset configuration system is working!")
        
    except Exception as e:
        print(f"❌ Error testing configuration: {e}")
        sys.exit(1)
