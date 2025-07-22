#!/usr/bin/env python3
"""
# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com
Deribit Asset Discovery Script

This script discovers all available underlying assets (currencies) from Deribit
that support options trading and stores them in a configuration file.

Usage:
    python discover_deribit_assets.py                    # Discover and save to config
    python discover_deribit_assets.py --output config.json  # Custom output file
    python discover_deribit_assets.py --test             # Test mode, don't save
    python discover_deribit_assets.py --verbose          # Detailed output
"""

import sys
import json
import requests
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Any, Optional
import time

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

class DeribitAssetDiscovery:
    """
    Discovers available underlying assets from Deribit exchange.
    """
    
    def __init__(self, timeout: int = 30, max_retries: int = 3):
        """
        Initialize asset discovery.
        
        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.base_url = "https://www.deribit.com/api/v2"
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Configure session headers
        self.session.headers.update({
            'User-Agent': 'Bitcoin-Options-Analytics-AssetDiscovery/1.0.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })

    def make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Optional[Dict]:
        """
        Make API request with retry logic.
        
        Args:
            endpoint: API endpoint (without base URL)
            params: Request parameters
            
        Returns:
            API response data or None if failed
        """
        url = f"{self.base_url}/{endpoint}"
        
        for attempt in range(self.max_retries):
            try:
                self.logger.debug(f"Making request to {endpoint} (attempt {attempt + 1})")
                
                response = self.session.get(url, params=params or {}, timeout=self.timeout)
                response.raise_for_status()
                
                data = response.json()
                
                if 'error' in data:
                    self.logger.error(f"API error: {data['error']}")
                    return None
                
                return data
                
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Request attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    self.logger.error(f"All {self.max_retries} attempts failed for {endpoint}")
                    
        return None

    def get_available_currencies(self) -> List[str]:
        """
        Get all currencies supported by Deribit.
        
        Returns:
            List of currency symbols
        """
        self.logger.info("Fetching available currencies from Deribit...")
        
        response = self.make_request("public/get_currencies")
        
        if not response or 'result' not in response:
            self.logger.error("Failed to fetch currencies")
            return []
        
        currencies = []
        for currency_info in response['result']:
            currency = currency_info.get('currency', '')
            if currency:
                currencies.append(currency)
                
        self.logger.info(f"Found {len(currencies)} currencies: {currencies}")
        return currencies

    def check_options_support(self, currency: str) -> Dict[str, Any]:
        """
        Check if a currency supports options trading and get details.
        
        Args:
            currency: Currency symbol to check
            
        Returns:
            Dictionary with options support info
        """
        self.logger.debug(f"Checking options support for {currency}")
        
        # Get instruments for this currency
        response = self.make_request("public/get_instruments", {
            "currency": currency,
            "kind": "option"
        })
        
        if not response or 'result' not in response:
            return {
                'currency': currency,
                'supports_options': False,
                'instruments_count': 0,
                'error': 'Failed to fetch instruments'
            }
        
        instruments = response['result']
        active_instruments = [inst for inst in instruments if inst.get('is_active', False)]
        
        # Get additional details
        details = {
            'currency': currency,
            'supports_options': len(instruments) > 0,
            'total_instruments': len(instruments),
            'active_instruments': len(active_instruments),
            'instruments_count': len(active_instruments),  # For backward compatibility
        }
        
        if active_instruments:
            # Get some sample instrument details
            sample_instrument = active_instruments[0]
            details.update({
                'sample_instrument': sample_instrument.get('instrument_name', ''),
                'contract_size': sample_instrument.get('contract_size', 1),
                'tick_size': sample_instrument.get('tick_size', 0.0001),
                'min_trade_amount': sample_instrument.get('min_trade_amount', 0.1),
            })
            
            # Get expiration dates
            expiration_dates = set()
            for inst in active_instruments:
                exp_date = inst.get('expiration_timestamp')
                if exp_date:
                    expiration_dates.add(exp_date)
            
            details['expiration_dates_count'] = len(expiration_dates)
            
            # Get strike price range
            strikes = []
            for inst in active_instruments:
                strike = inst.get('strike')
                if strike:
                    strikes.append(strike)
            
            if strikes:
                details.update({
                    'min_strike': min(strikes),
                    'max_strike': max(strikes),
                    'strike_count': len(set(strikes))
                })
        
        return details

    def get_index_prices(self, currencies: List[str]) -> Dict[str, Optional[float]]:
        """
        Get current index prices for currencies.
        
        Args:
            currencies: List of currency symbols
            
        Returns:
            Dictionary mapping currency to current price
        """
        self.logger.info("Fetching current index prices...")
        
        prices = {}
        
        for currency in currencies:
            try:
                response = self.make_request("public/get_index_price", {
                    "index_name": f"{currency}_usd"
                })
                
                if response and 'result' in response:
                    price = response['result'].get('index_price')
                    prices[currency] = price
                    self.logger.debug(f"{currency} price: ${price:,.2f}" if price else f"{currency}: No price")
                else:
                    prices[currency] = None
                    
            except Exception as e:
                self.logger.warning(f"Failed to get price for {currency}: {e}")
                prices[currency] = None
                
        return prices

    def discover_all_assets(self) -> Dict[str, Any]:
        """
        Discover all available assets and their options trading capabilities.
        
        Returns:
            Complete asset discovery results
        """
        self.logger.info("🔍 Starting comprehensive asset discovery...")
        
        discovery_start = datetime.now()
        
        # Step 1: Get all currencies
        currencies = self.get_available_currencies()
        
        if not currencies:
            self.logger.error("No currencies found, aborting discovery")
            return {
                'discovery_time': discovery_start.isoformat(),
                'success': False,
                'error': 'Failed to fetch currencies'
            }
        
        # Step 2: Check options support for each currency
        self.logger.info(f"Checking options support for {len(currencies)} currencies...")
        
        assets_info = {}
        options_enabled_assets = []
        
        for i, currency in enumerate(currencies, 1):
            self.logger.info(f"Checking {currency} ({i}/{len(currencies)})...")
            
            options_info = self.check_options_support(currency)
            assets_info[currency] = options_info
            
            if options_info['supports_options'] and options_info['instruments_count'] > 0:
                options_enabled_assets.append(currency)
                self.logger.info(f"✅ {currency}: {options_info['instruments_count']} active options")
            else:
                self.logger.debug(f"❌ {currency}: No options trading")
            
            # Small delay to be respectful to API
            time.sleep(0.1)
        
        # Step 3: Get current prices for options-enabled assets
        prices = {}
        if options_enabled_assets:
            prices = self.get_index_prices(options_enabled_assets)
        
        # Step 4: Compile final results
        discovery_end = datetime.now()
        discovery_duration = (discovery_end - discovery_start).total_seconds()
        
        results = {
            'discovery_time': discovery_start.isoformat(),
            'discovery_duration_seconds': discovery_duration,
            'success': True,
            'total_currencies_checked': len(currencies),
            'options_enabled_currencies': options_enabled_assets,
            'options_enabled_count': len(options_enabled_assets),
            'all_currencies': currencies,
            'current_prices': prices,
            'detailed_info': assets_info,
            'recommended_for_trading': [],
            'metadata': {
                'api_base_url': self.base_url,
                'discovery_script_version': '1.0.0',
                'last_updated': discovery_end.isoformat()
            }
        }
        
        # Step 5: Determine recommended assets for trading
        recommended = []
        for currency in options_enabled_assets:
            info = assets_info[currency]
            price = prices.get(currency)
            
            # Criteria for recommendation
            if (info['instruments_count'] >= 10 and  # Sufficient options
                price and price > 1 and  # Reasonable price
                info.get('strike_count', 0) >= 5):  # Good strike range
                recommended.append(currency)
        
        results['recommended_for_trading'] = recommended
        
        # Log summary
        self.logger.info("🎯 Asset Discovery Complete!")
        self.logger.info(f"   Total currencies: {len(currencies)}")
        self.logger.info(f"   Options enabled: {len(options_enabled_assets)} - {options_enabled_assets}")
        self.logger.info(f"   Recommended: {len(recommended)} - {recommended}")
        self.logger.info(f"   Discovery time: {discovery_duration:.1f} seconds")
        
        return results

    def save_config(self, 
                   discovery_results: Dict[str, Any], 
                   config_file: str = "config/assets_config.json",
                   create_dirs: bool = True) -> bool:
        """
        Save discovery results to configuration file.
        
        Args:
            discovery_results: Results from discover_all_assets()
            config_file: Path to save configuration
            create_dirs: Whether to create directories if they don't exist
            
        Returns:
            True if saved successfully
        """
        config_path = Path(config_file)
        
        try:
            # Create directories if needed
            if create_dirs:
                config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create a clean config structure
            config = {
                'assets': {
                    'all_currencies': discovery_results['all_currencies'],
                    'options_enabled': discovery_results['options_enabled_currencies'],
                    'recommended': discovery_results['recommended_for_trading'],
                    'default_currencies': discovery_results['recommended_for_trading'][:3] if discovery_results['recommended_for_trading'] else ['BTC'],  # Top 3 or fallback
                },
                'prices': discovery_results['current_prices'],
                'details': discovery_results['detailed_info'],
                'metadata': discovery_results['metadata'],
                'discovery_info': {
                    'last_discovery': discovery_results['discovery_time'],
                    'duration_seconds': discovery_results['discovery_duration_seconds'],
                    'total_checked': discovery_results['total_currencies_checked'],
                    'success': discovery_results['success']
                }
            }
            
            # Save to file
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2, sort_keys=True)
            
            self.logger.info(f"✅ Configuration saved to {config_path}")
            self.logger.info(f"   Default currencies: {config['assets']['default_currencies']}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            return False


def main():
    """Main script entry point."""
    parser = argparse.ArgumentParser(
        description="Discover Deribit underlying assets for options trading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python discover_deribit_assets.py                      # Discover and save to default config
  python discover_deribit_assets.py --test               # Test mode, don't save
  python discover_deribit_assets.py --verbose            # Detailed logging
  python discover_deribit_assets.py --output my_config.json  # Custom output file
  python discover_deribit_assets.py --timeout 60         # Longer timeout
        """
    )
    
    parser.add_argument(
        '--output', '-o',
        default='config/assets_config.json',
        help='Output configuration file path (default: config/assets_config.json)'
    )
    
    parser.add_argument(
        '--test', '-t',
        action='store_true',
        help='Test mode - discover but do not save configuration'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=30,
        help='Request timeout in seconds (default: 30)'
    )
    
    parser.add_argument(
        '--max-retries',
        type=int,
        default=3,
        help='Maximum retry attempts (default: 3)'
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("🚀 Deribit Asset Discovery Tool")
    print("=" * 50)
    
    try:
        # Initialize discovery
        discovery = DeribitAssetDiscovery(
            timeout=args.timeout,
            max_retries=args.max_retries
        )
        
        # Run discovery
        results = discovery.discover_all_assets()
        
        if not results['success']:
            print("❌ Asset discovery failed")
            return 1
        
        # Print summary
        print("\n📊 DISCOVERY SUMMARY:")
        print("-" * 30)
        print(f"Total currencies: {results['total_currencies_checked']}")
        print(f"Options enabled: {results['options_enabled_count']}")
        print(f"Recommended: {len(results['recommended_for_trading'])}")
        print(f"Discovery time: {results['discovery_duration_seconds']:.1f} seconds")
        
        print(f"\n✅ Options-enabled currencies:")
        for currency in results['options_enabled_currencies']:
            info = results['detailed_info'][currency]
            price = results['current_prices'].get(currency)
            price_str = f"${price:,.2f}" if price else "N/A"
            print(f"   {currency}: {info['instruments_count']} options, price: {price_str}")
        
        print(f"\n🎯 Recommended for trading:")
        for currency in results['recommended_for_trading']:
            info = results['detailed_info'][currency]
            price = results['current_prices'].get(currency)
            price_str = f"${price:,.2f}" if price else "N/A"
            print(f"   {currency}: {info['instruments_count']} options, {info.get('strike_count', 0)} strikes, price: {price_str}")
        
        # Save configuration unless in test mode
        if not args.test:
            success = discovery.save_config(results, args.output)
            if success:
                print(f"\n✅ Configuration saved to: {args.output}")
                print("🎯 You can now use these currencies in your trading system!")
            else:
                print("\n❌ Failed to save configuration")
                return 1
        else:
            print("\n🧪 Test mode - configuration not saved")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Discovery interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        logging.exception("Discovery failed with exception")
        return 1


if __name__ == "__main__":
    sys.exit(main())
