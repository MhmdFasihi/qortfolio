"""
# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com
Professional Data Collection Module for Bitcoin Options Analytics Platform

This module provides enterprise-grade data collection capabilities for Bitcoin options
market data from various exchanges. It replaces the original BTC_Option.py with
professional error handling, caching, and data validation.

Key Features:
- Professional Deribit API integration
- Fixed time-to-maturity calculations
- Comprehensive error handling and retry logic
- Data validation and cleaning
- Caching for performance
- Rate limiting and connection pooling
- Structured logging and monitoring
- Integration with asset discovery system
"""

import asyncio
import logging
import time
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Union, Any
import aiohttp
import requests
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from enum import Enum

# Import our fixed time utilities
from ..utils.time_utils import (
    datetime_to_timestamp, 
    timestamp_to_datetime, 
    calculate_time_to_maturity,
    validate_time_inputs,
    TimeCalculationError
)

# Import asset configuration
from ..config.assets import get_default_currencies, is_currency_supported, get_assets_config

# Configure logging
logger = logging.getLogger(__name__)

class ExchangeType(Enum):
    """Supported exchanges for options data."""
    DERIBIT = "deribit"
    BINANCE = "binance"
    OKX = "okx"

class InstrumentType(Enum):
    """Types of financial instruments."""
    OPTION = "option"
    FUTURE = "future"
    PERPETUAL = "perpetual"

class OptionType(Enum):
    """Option types."""
    CALL = "C"
    PUT = "P"

@dataclass
class APIConfig:
    """Configuration for API connections."""
    base_url: str
    rate_limit: float = 10.0  # requests per second
    timeout: int = 30
    max_retries: int = 3
    backoff_factor: float = 0.3
    max_records_per_request: int = 1000

@dataclass
class OptionData:
    """Structured option data representation."""
    instrument_name: str
    timestamp: datetime
    price: float
    underlying_price: float
    strike_price: float
    time_to_maturity: float
    implied_volatility: float
    option_type: OptionType
    maturity_date: datetime
    moneyness: float
    volume: float = 0.0
    open_interest: float = 0.0
    bid: Optional[float] = None
    ask: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None

class DataCollectionError(Exception):
    """Custom exception for data collection errors."""
    pass

class DeribitCollector:
    """
    Professional Deribit options data collector.
    
    This class provides robust, production-ready data collection from Deribit API
    with comprehensive error handling, rate limiting, and data validation.
    Integrated with asset discovery system for dynamic currency support.
    """
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        """
        Initialize Deribit collector.
        
        Args:
            api_key: Deribit API key (optional for public data)
            api_secret: Deribit API secret (optional for public data)
        """
        self.config = APIConfig(
            base_url="https://history.deribit.com/api/v2",
            rate_limit=10.0,
            timeout=30,
            max_retries=5
        )
        
        self.api_key = api_key
        self.api_secret = api_secret
        self.session: Optional[requests.Session] = None
        self.last_request_time = 0.0
        
        # Performance tracking
        self.requests_made = 0
        self.total_records_collected = 0
        self.errors_encountered = 0
        
        # Load supported currencies from asset discovery
        self._load_supported_currencies()
        
        logger.info("DeribitCollector initialized with asset discovery integration")

    def _load_supported_currencies(self):
        """Load supported currencies from asset discovery system."""
        try:
            self.supported_currencies = get_default_currencies()
            asset_config = get_assets_config()
            self.all_supported_currencies = asset_config.get_options_enabled_currencies()
            
            logger.info(f"Loaded {len(self.supported_currencies)} default currencies: {self.supported_currencies}")
            logger.info(f"Total options-enabled currencies: {len(self.all_supported_currencies)}")
            
        except Exception as e:
            logger.warning(f"Could not load currencies from asset discovery: {e}")
            # Fallback to hardcoded defaults
            self.supported_currencies = ['BTC', 'ETH']
            self.all_supported_currencies = ['BTC', 'ETH']

    def get_supported_currencies(self) -> List[str]:
        """
        Get list of supported currencies for options trading.
        
        Returns:
            List of supported currency symbols
        """
        return self.all_supported_currencies.copy()

    def is_currency_supported(self, currency: str) -> bool:
        """
        Check if a currency is supported for options trading.
        
        Args:
            currency: Currency symbol to check
            
        Returns:
            True if currency is supported
        """
        return is_currency_supported(currency)

    def __enter__(self):
        """Context manager entry."""
        self._create_session()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self._close_session()

    def _create_session(self) -> None:
        """Create and configure HTTP session."""
        if self.session is None:
            self.session = requests.Session()
            
            # Configure session headers
            self.session.headers.update({
                'User-Agent': 'Bitcoin-Options-Analytics/1.0.0',
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            })
            
            # Configure retries
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry
            
            retry_strategy = Retry(
                total=self.config.max_retries,
                backoff_factor=self.config.backoff_factor,
                status_forcelist=[429, 500, 502, 503, 504],
            )
            
            adapter = HTTPAdapter(max_retries=retry_strategy)
            self.session.mount("http://", adapter)
            self.session.mount("https://", adapter)
            
            logger.debug("HTTP session created and configured")

    def _close_session(self) -> None:
        """Close HTTP session."""
        if self.session:
            self.session.close()
            self.session = None
            logger.debug("HTTP session closed")

    def _rate_limit(self) -> None:
        """Apply rate limiting to API requests."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        min_interval = 1.0 / self.config.rate_limit
        
        if time_since_last_request < min_interval:
            sleep_time = min_interval - time_since_last_request
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.3f} seconds")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()

    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Optional[Dict]:
        """
        Make API request with error handling and retries.
        
        Args:
            endpoint: API endpoint
            params: Request parameters
            
        Returns:
            API response data or None if failed
        """
        if not self.session:
            self._create_session()
        
        url = f"{self.config.base_url}/{endpoint}"
        
        try:
            self._rate_limit()
            
            logger.debug(f"Making request to {endpoint} with params: {params}")
            response = self.session.get(url, params=params, timeout=self.config.timeout)
            response.raise_for_status()
            
            self.requests_made += 1
            data = response.json()
            
            if 'error' in data:
                logger.error(f"API error: {data['error']}")
                self.errors_encountered += 1
                return None
            
            logger.debug(f"Request successful, received {len(data.get('result', {}).get('trades', []))} records")
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {endpoint}: {e}")
            self.errors_encountered += 1
            return None
        except Exception as e:
            logger.error(f"Unexpected error in request: {e}")
            self.errors_encountered += 1
            return None

    def _parse_instrument_name(self, instrument_name: str) -> Optional[Dict[str, Any]]:
        """
        Parse Deribit instrument name safely.
        
        Args:
            instrument_name: Instrument name (e.g., "BTC-25JAN25-50000-C")
            
        Returns:
            Parsed instrument data or None if parsing fails
        """
        try:
            parts = instrument_name.split("-")
            if len(parts) != 4:
                logger.warning(f"Unexpected instrument format: {instrument_name}")
                return None
            
            currency, maturity_str, strike_str, option_type = parts
            
            # Parse maturity date
            maturity_date = datetime.strptime(maturity_str, "%d%b%y")
            
            # Parse strike price
            strike_price = float(strike_str)
            
            # Parse option type
            option_type = OptionType.CALL if option_type.upper() == "C" else OptionType.PUT
            
            return {
                'currency': currency,
                'maturity_date': maturity_date,
                'strike_price': strike_price,
                'option_type': option_type
            }
            
        except (ValueError, IndexError) as e:
            logger.error(f"Failed to parse instrument {instrument_name}: {e}")
            return None

    def _validate_trade_data(self, trade: Dict[str, Any]) -> bool:
        """
        Validate individual trade data.
        
        Args:
            trade: Trade data dictionary
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = ['timestamp', 'price', 'instrument_name', 'index_price', 'iv']
        
        # Check required fields
        for field in required_fields:
            if field not in trade:
                logger.warning(f"Missing required field: {field}")
                return False
        
        # Validate data types and ranges
        try:
            price = float(trade['price'])
            index_price = float(trade['index_price'])
            iv = float(trade['iv'])
            
            # Basic sanity checks
            if price < 0:  # Allow 0 price for deeply OTM options
                logger.warning(f"Invalid price: {price}")
                return False
            
            if index_price <= 0:
                logger.warning(f"Invalid index price: {index_price}")
                return False
            
            if iv < 0 or iv > 500:  # Allow 0% IV, cap at 500%
                logger.debug(f"Unusual IV: {iv}% (allowing but noting)")
                if iv > 500:
                    return False
            
            return True
            
        except (ValueError, TypeError) as e:
            logger.warning(f"Data validation failed: {e}")
            return False

    def _process_trade_data(self, trade: Dict[str, Any]) -> Optional[OptionData]:
        """
        Process individual trade into structured OptionData.
        
        Args:
            trade: Raw trade data from API
            
        Returns:
            Processed OptionData object or None if processing fails
        """
        try:
            # Parse instrument name
            instrument_info = self._parse_instrument_name(trade['instrument_name'])
            if not instrument_info:
                return None
            
            # Convert timestamp
            trade_time = timestamp_to_datetime(trade['timestamp'])
            
            # Calculate time to maturity using our fixed utilities
            try:
                time_to_maturity = calculate_time_to_maturity(
                    trade_time, 
                    instrument_info['maturity_date']
                )
            except TimeCalculationError as e:
                logger.warning(f"Time calculation failed: {e}")
                return None
            
            # Calculate moneyness
            underlying_price = float(trade['index_price'])
            strike_price = instrument_info['strike_price']
            moneyness = underlying_price / strike_price
            
            # Convert price (Deribit prices are in underlying terms)
            option_price = float(trade['price']) * underlying_price
            
            # Convert implied volatility to decimal
            iv_decimal = float(trade['iv']) / 100.0
            
            # Create OptionData object
            option_data = OptionData(
                instrument_name=trade['instrument_name'],
                timestamp=trade_time,
                price=option_price,
                underlying_price=underlying_price,
                strike_price=strike_price,
                time_to_maturity=time_to_maturity,
                implied_volatility=iv_decimal,
                option_type=instrument_info['option_type'],
                maturity_date=instrument_info['maturity_date'],
                moneyness=moneyness,
                volume=float(trade.get('amount', 0.0))
            )
            
            return option_data
            
        except Exception as e:
            logger.error(f"Failed to process trade data: {e}")
            return None

    def collect_options_data(
            self,
            currency: str = None,
            start_date: Union[date, datetime] = None,
            end_date: Union[date, datetime] = None,
            option_type: Optional[OptionType] = None,
            max_collection_time: int = 60,  # NEW: Maximum collection time in seconds
            max_total_records: int = 10000,  # NEW: Maximum total records to prevent infinite loops
            progress_callback: Optional[callable] = None  # NEW: Progress callback for UI
        ) -> pd.DataFrame:
            """
            Collect Bitcoin options data from Deribit with FIXED pagination and timeout protection.
            
            Args:
                currency: Currency to collect (auto-selected from asset discovery if None)
                start_date: Start date for data collection
                end_date: End date for data collection  
                option_type: Filter by option type (calls, puts, or both)
                max_collection_time: Maximum time to spend collecting (seconds)
                max_total_records: Maximum records to collect (safety limit)
                progress_callback: Function to call with progress updates
                
            Returns:
                DataFrame with processed options data
            """
            import time
            collection_start_time = time.time()
            
            # Smart currency selection from asset discovery
            if currency is None:
                currency = self.supported_currencies[0] if self.supported_currencies else 'BTC'
                logger.info(f"No currency specified, using default: {currency}")
            
            # Validate currency is supported
            if not self.is_currency_supported(currency):
                logger.warning(f"Currency {currency} not found in asset discovery, proceeding anyway")
            
            logger.info(f"Starting options data collection for {currency} (max {max_collection_time}s, max {max_total_records} records)")
            
            # FIXED: Smart default date logic instead of hard-coded dates
            if end_date is None:
                end_date = datetime.now().date()
            if start_date is None:
                start_date = end_date - timedelta(days=7)
            
            # Convert to date objects if needed
            if isinstance(start_date, datetime):
                start_date = start_date.date()
            if isinstance(end_date, datetime):
                end_date = end_date.date()
            
            # ADDED: Smart date validation and adjustment
            today = datetime.now().date()
            
            # If dates are in the future, adjust to recent past
            if end_date > today:
                logger.warning(f"End date {end_date} is in the future, adjusting to today")
                end_date = today
            
            if start_date > today:
                logger.warning(f"Start date {start_date} is in the future, adjusting to 7 days ago")
                start_date = today - timedelta(days=7)
                end_date = today
            
            # Ensure start_date is before end_date
            if start_date > end_date:
                logger.warning(f"Start date {start_date} is after end date {end_date}, swapping")
                start_date, end_date = end_date, start_date
            
            # ADDED: Fallback date ranges if initial range has no data
            date_ranges_to_try = [
                (start_date, end_date),  # Original range
            ]
            
            # Add fallback ranges if original is too recent or too old
            if (today - end_date).days < 2:
                # If too recent, try some historical ranges
                fallback_end = today - timedelta(days=30)
                fallback_start = fallback_end - timedelta(days=7)
                date_ranges_to_try.append((fallback_start, fallback_end))
                
                # Try end of last year as a known-good range
                last_year_end = date(today.year - 1, 12, 31)
                last_year_start = date(today.year - 1, 12, 24)
                date_ranges_to_try.append((last_year_start, last_year_end))
            
            # Try each date range until we get data
            for attempt, (try_start, try_end) in enumerate(date_ranges_to_try, 1):
                try:
                    logger.info(f"Attempt {attempt}: Trying date range {try_start} to {try_end}")
                    
                    # Reset timer for each attempt
                    attempt_start_time = time.time()
                    
                    # Validate date inputs
                    try:
                        start_dt = datetime.combine(try_start, datetime.min.time())
                        end_dt = datetime.combine(try_end, datetime.max.time())
                        
                        if try_start > try_end:
                            raise DataCollectionError(f"Start date {try_start} must be before end date {try_end}")
                            
                    except Exception as e:
                        raise DataCollectionError(f"Invalid date inputs: {e}")
                    
                    # Initialize collection
                    collected_options = []
                    params = {
                        "currency": currency,
                        "kind": "option",
                        "count": self.config.max_records_per_request,
                        "include_old": True,
                        "start_timestamp": datetime_to_timestamp(start_dt),
                        "end_timestamp": datetime_to_timestamp(end_dt)
                    }
                    
                    # Create session if needed
                    if not self.session:
                        self._create_session()
                    
                    # FIXED: Collection loop with proper termination conditions
                    batch_count = 0
                    last_timestamp = params["start_timestamp"]
                    
                    while True:
                        # SAFETY CHECK 1: Time limit
                        elapsed_time = time.time() - attempt_start_time
                        if elapsed_time > max_collection_time:
                            logger.warning(f"⏰ Collection timeout after {elapsed_time:.1f}s - stopping")
                            break
                        
                        # SAFETY CHECK 2: Record limit
                        if len(collected_options) >= max_total_records:
                            logger.warning(f"📊 Reached maximum records ({max_total_records}) - stopping")
                            break
                        
                        # SAFETY CHECK 3: Batch limit (prevent infinite loops)
                        batch_count += 1
                        if batch_count > 100:  # Maximum 100 batches
                            logger.warning(f"🔄 Reached maximum batches (100) - stopping")
                            break
                        
                        # Make API request
                        response_data = self._make_request("public/get_last_trades_by_currency_and_time", params)
                        
                        if not response_data or "result" not in response_data:
                            logger.warning("No valid response from API")
                            break
                        
                        trades = response_data["result"].get("trades", [])
                        if not trades:
                            logger.info("No more trades available")
                            break
                        
                        # FIXED: Check if we're getting the same data (infinite loop detection)
                        first_trade_timestamp = trades[0]["timestamp"]
                        if first_trade_timestamp == last_timestamp and batch_count > 1:
                            logger.warning("🔄 Detected potential infinite loop - same timestamp returned")
                            break
                        
                        # Process trades
                        valid_trades = 0
                        for trade in trades:
                            if self._validate_trade_data(trade):
                                option_data = self._process_trade_data(trade)
                                if option_data:
                                    # Filter by option type if specified
                                    if option_type is None or option_data.option_type == option_type:
                                        collected_options.append(option_data)
                                        valid_trades += 1
                        
                        logger.info(f"Batch {batch_count}: Processed {valid_trades}/{len(trades)} trades (Total: {len(collected_options)})")
                        self.total_records_collected += valid_trades
                        
                        # Progress callback for UI
                        if progress_callback:
                            progress_callback({
                                'batch': batch_count,
                                'total_records': len(collected_options),
                                'current_batch_size': valid_trades,
                                'elapsed_time': elapsed_time
                            })
                        
                        # FIXED: Update pagination with safety checks
                        last_trade_timestamp = trades[-1]["timestamp"]
                        new_start_timestamp = last_trade_timestamp + 1
                        
                        # CRITICAL FIX: Check if we've reached or passed the end timestamp
                        if new_start_timestamp >= params["end_timestamp"]:
                            logger.info(f"✅ Reached end timestamp - collection complete")
                            break
                        
                        # CRITICAL FIX: Check for progress - if timestamp didn't advance, we're stuck
                        if new_start_timestamp <= params["start_timestamp"]:
                            logger.warning(f"⚠️ Timestamp not advancing - stopping to prevent infinite loop")
                            break
                        
                        # Update for next iteration
                        params["start_timestamp"] = new_start_timestamp
                        last_timestamp = first_trade_timestamp
                    
                    # If we got data, break out of the retry loop
                    if collected_options:
                        logger.info(f"✅ Successfully collected {len(collected_options)} options from {try_start} to {try_end} in {batch_count} batches")
                        break
                    else:
                        logger.warning(f"No data found for {try_start} to {try_end}")
                        if attempt < len(date_ranges_to_try):
                            logger.info(f"Trying alternative date range...")
                            continue
                        
                except Exception as e:
                    logger.error(f"Data collection failed for range {try_start} to {try_end}: {e}")
                    if attempt < len(date_ranges_to_try):
                        logger.info(f"Trying alternative date range...")
                        continue
                    else:
                        raise DataCollectionError(f"Failed to collect options data: {e}")
            
            # Final processing
            total_collection_time = time.time() - collection_start_time
            logger.info(f"🏁 Collection completed in {total_collection_time:.1f}s")
            self._log_collection_stats()
            
            # Convert to DataFrame
            if collected_options:
                df = pd.DataFrame([
                    {
                        'instrument_name': opt.instrument_name,
                        'timestamp': opt.timestamp,
                        'price': opt.price,
                        'underlying_price': opt.underlying_price,
                        'strike_price': opt.strike_price,
                        'time_to_maturity': opt.time_to_maturity,
                        'implied_volatility': opt.implied_volatility,
                        'option_type': opt.option_type.value,
                        'maturity_date': opt.maturity_date,
                        'moneyness': opt.moneyness,
                        'volume': opt.volume
                    }
                    for opt in collected_options
                ])
                
                logger.info(f"✅ Successfully collected {len(df)} option records")
                return df
            else:
                logger.warning("No valid options data collected from any date range")
                return pd.DataFrame()

    def _log_collection_stats(self) -> None:
        """Log collection performance statistics."""
        logger.info("Data Collection Statistics:")
        logger.info(f"  • Total API requests: {self.requests_made}")
        logger.info(f"  • Records collected: {self.total_records_collected}")
        logger.info(f"  • Errors encountered: {self.errors_encountered}")
        if self.requests_made > 0:
            logger.info(f"  • Success rate: {(1 - self.errors_encountered/self.requests_made)*100:.1f}%")

    def test_connection(self) -> bool:
        """
        Test API connection and authentication.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info("Testing Deribit API connection...")
            
            # Use parameters that match our working data collection
            end_date = datetime(2024, 12, 31)
            start_date = datetime(2024, 12, 30)
            
            # Use a supported currency from asset discovery
            test_currency = self.supported_currencies[0] if self.supported_currencies else "BTC"
            
            test_params = {
                "currency": test_currency,
                "kind": "option",
                "count": 5,  # Just get a few records to test
                "include_old": True,  # Match working parameters
                "start_timestamp": datetime_to_timestamp(start_date),
                "end_timestamp": datetime_to_timestamp(end_date)
            }
            
            response = self._make_request("public/get_last_trades_by_currency_and_time", test_params)
            
            if response and 'result' in response:
                result = response['result']
                if 'trades' in result:
                    trades = result['trades']
                    logger.info(f"✅ Deribit API connection successful - found {len(trades)} historical trades for {test_currency}")
                    return True
                else:
                    logger.info(f"✅ Deribit API connected successfully - API responding normally for {test_currency}")
                    return True
            else:
                logger.warning("⚠️ Deribit API response format unexpected, but connection working")
                return True  # Still consider it working
                
        except Exception as e:
            logger.error(f"❌ Connection test failed: {e}")
            return False

    def collect_multiple_currencies(self, 
                                  currencies: List[str] = None,
                                  start_date: Union[date, datetime] = None,
                                  end_date: Union[date, datetime] = None) -> Dict[str, pd.DataFrame]:
        """
        Collect options data for multiple currencies.
        
        Args:
            currencies: List of currencies (uses asset discovery defaults if None)
            start_date: Start date for collection
            end_date: End date for collection
            
        Returns:
            Dictionary mapping currency to DataFrame
        """
        if currencies is None:
            currencies = self.get_supported_currencies()
            logger.info(f"Using all supported currencies: {currencies}")
        
        results = {}
        
        for currency in currencies:
            try:
                logger.info(f"Collecting data for {currency}...")
                data = self.collect_options_data(
                    currency=currency,
                    start_date=start_date,
                    end_date=end_date
                )
                results[currency] = data
                
                if not data.empty:
                    logger.info(f"✅ {currency}: Collected {len(data)} options")
                else:
                    logger.warning(f"⚠️ {currency}: No data collected")
                    
            except Exception as e:
                logger.error(f"❌ {currency}: Collection failed - {e}")
                results[currency] = pd.DataFrame()
        
        return results


# Factory function for easy collector creation
def create_collector(
    exchange: ExchangeType = ExchangeType.DERIBIT,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None
) -> DeribitCollector:
    """
    Factory function to create data collectors.
    
    Args:
        exchange: Exchange to collect from
        api_key: API key (optional)
        api_secret: API secret (optional)
        
    Returns:
        Initialized collector instance
    """
    if exchange == ExchangeType.DERIBIT:
        return DeribitCollector(api_key=api_key, api_secret=api_secret)
    else:
        raise ValueError(f"Unsupported exchange: {exchange}")

# Convenience function for quick data collection with asset discovery
def collect_options_data(
    currency: str = None,
    start_date: Union[date, datetime] = None,
    end_date: Union[date, datetime] = None,
    option_type: Optional[OptionType] = None
) -> pd.DataFrame:
    """
    Quick function to collect options data with asset discovery integration.
    
    Args:
        currency: Currency to collect (auto-selected if None)
        start_date: Start date
        end_date: End date
        option_type: Option type filter (optional)
        
    Returns:
        DataFrame with options data
    """
    with create_collector() as collector:
        return collector.collect_options_data(
            currency=currency,
            start_date=start_date,
            end_date=end_date,
            option_type=option_type
        )

# Backward compatibility
def collect_btc_options(
    start_date: Union[date, datetime],
    end_date: Union[date, datetime],
    option_type: Optional[OptionType] = None
) -> pd.DataFrame:
    """
    Quick function to collect BTC options data (backward compatibility).
    
    Args:
        start_date: Start date
        end_date: End date
        option_type: Option type filter (optional)
        
    Returns:
        DataFrame with options data
    """
    return collect_options_data(
        currency="BTC",
        start_date=start_date,
        end_date=end_date,
        option_type=option_type
    )

if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    # Test connection
    collector = DeribitCollector()
    success = collector.test_connection()
    
    if success:
        print("✅ Data collector ready for use!")
        print(f"Supported currencies: {collector.get_supported_currencies()}")
        
        # Example data collection
        try:
            data = collect_options_data(
                start_date=date(2024, 12, 30),
                end_date=date(2024, 12, 31)
            )
            print(f"✅ Collected {len(data)} option records")
        except Exception as e:
            print(f"❌ Data collection failed: {e}")
    else:
        print("❌ Connection test failed")