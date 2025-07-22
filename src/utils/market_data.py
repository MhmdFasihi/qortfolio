"""
# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com
Market Data Helper Functions for Real-Time Defaults

This module provides functions to fetch current market data for setting
intelligent defaults in the dashboard interface.
"""

import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List
import streamlit as st
from functools import lru_cache
import time

logger = logging.getLogger(__name__)

class MarketDataError(Exception):
    """Custom exception for market data errors."""
    pass

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_current_crypto_price(currency: str = "BTC") -> Optional[float]:
    """
    Get current cryptocurrency price from CoinGecko API.
    
    Args:
        currency: Currency symbol (BTC, ETH, etc.)
        
    Returns:
        Current price in USD or None if failed
    """
    try:
        # CoinGecko API endpoint (free, no API key required)
        currency_map = {
            'BTC': 'bitcoin',
            'ETH': 'ethereum'
        }
        
        coin_id = currency_map.get(currency.upper(), 'bitcoin')
        url = f"https://api.coingecko.com/api/v3/simple/price"
        
        params = {
            'ids': coin_id,
            'vs_currencies': 'usd'
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        price = data[coin_id]['usd']
        
        logger.info(f"✅ Current {currency} price: ${price:,.2f}")
        return float(price)
        
    except Exception as e:
        logger.warning(f"⚠️ Could not fetch {currency} price: {e}")
        
        # Fallback prices (reasonable estimates)
        fallback_prices = {
            'BTC': 83000.0,
            'ETH': 2600.0
        }
        
        fallback_price = fallback_prices.get(currency.upper(), 30000.0)
        logger.info(f"Using fallback price for {currency}: ${fallback_price:,.2f}")
        return fallback_price

@st.cache_data(ttl=600)  # Cache for 10 minutes  
def get_available_options_data(currency: str = "BTC") -> Optional[pd.DataFrame]:
    """
    Get a sample of available options data to determine strikes and maturities.
    
    Args:
        currency: Currency to fetch options for
        
    Returns:
        DataFrame with available options or None if failed
    """
    try:
        from ..data.collectors import DeribitCollector
        from datetime import date, timedelta
        
        # Use recent historical data (last week) to get available options
        end_date = date.today() - timedelta(days=1)  # Yesterday
        start_date = end_date - timedelta(days=2)    # Day before yesterday
        
        logger.info(f"🔍 Fetching available {currency} options for defaults...")
        
        with DeribitCollector() as collector:
            # Quick collection with limits to avoid hanging
            data = collector.collect_options_data(
                currency=currency,
                start_date=start_date,
                end_date=end_date,
                max_collection_time=15,  # Maximum 15 seconds
                max_total_records=1000   # Limit records
            )
            
            if not data.empty:
                logger.info(f"✅ Found {len(data)} {currency} options for analysis")
                return data
            else:
                logger.warning(f"⚠️ No {currency} options data found")
                return None
                
    except Exception as e:
        logger.warning(f"⚠️ Could not fetch {currency} options data: {e}")
        return None

def get_nearest_strike_price(current_price: float, 
                           available_options: Optional[pd.DataFrame] = None,
                           currency: str = "BTC") -> float:
    """
    Get the nearest available strike price to current market price.
    
    Args:
        current_price: Current market price
        available_options: DataFrame with available options
        currency: Currency symbol
        
    Returns:
        Nearest strike price
    """
    try:
        if available_options is not None and not available_options.empty and 'strike_price' in available_options.columns:
            # Find the nearest strike price from actual options
            strikes = available_options['strike_price'].unique()
            nearest_strike = min(strikes, key=lambda x: abs(x - current_price))
            
            logger.info(f"✅ Nearest {currency} strike to ${current_price:,.0f}: ${nearest_strike:,.0f}")
            return float(nearest_strike)
        else:
            # Generate intelligent estimate based on current price
            # Round to nearest common option strike increment
            if current_price >= 50000:
                increment = 5000  # $5k increments for high prices
            elif current_price >= 20000:
                increment = 2000  # $2k increments for medium prices  
            else:
                increment = 1000  # $1k increments for lower prices
            
            # Find nearest increment above current price (typical for calls)
            nearest_strike = ((current_price // increment) + 1) * increment
            
            logger.info(f"📊 Estimated {currency} strike near ${current_price:,.0f}: ${nearest_strike:,.0f}")
            return float(nearest_strike)
            
    except Exception as e:
        logger.warning(f"⚠️ Could not determine nearest strike: {e}")
        # Safe fallback
        return current_price * 1.05  # 5% above current price

def get_nearest_maturity_days(available_options: Optional[pd.DataFrame] = None,
                            currency: str = "BTC") -> float:
    """
    Get the nearest available maturity in days.
    
    Args:
        available_options: DataFrame with available options
        currency: Currency symbol
        
    Returns:
        Days to nearest maturity
    """
    try:
        if available_options is not None and not available_options.empty:
            # Calculate days to maturity for all options
            today = datetime.now().date()
            
            if 'maturity_date' in available_options.columns:
                # Convert maturity dates to days from now
                maturities = pd.to_datetime(available_options['maturity_date']).dt.date
                days_to_maturity = [(maturity - today).days for maturity in maturities if maturity > today]
                
                if days_to_maturity:
                    nearest_days = min([days for days in days_to_maturity if days > 0])
                    logger.info(f"✅ Nearest {currency} maturity: {nearest_days} days")
                    return float(nearest_days)
            
            elif 'time_to_maturity' in available_options.columns:
                # Use time_to_maturity column (in years, convert to days)
                ttm_years = available_options['time_to_maturity']
                ttm_days = ttm_years * 365.25
                valid_days = ttm_days[ttm_days > 1]  # At least 1 day
                
                if not valid_days.empty:
                    nearest_days = valid_days.min()
                    logger.info(f"✅ Nearest {currency} maturity: {nearest_days:.0f} days")
                    return float(nearest_days)
        
        # Fallback to common option expiration periods
        common_periods = [7, 14, 30, 60, 90]  # 1 week, 2 weeks, 1 month, 2 months, 3 months
        fallback_days = 30.0  # Default to 30 days
        
        logger.info(f"📊 Using default {currency} maturity: {fallback_days} days")
        return fallback_days
        
    except Exception as e:
        logger.warning(f"⚠️ Could not determine nearest maturity: {e}")
        return 30.0  # Safe fallback

def get_smart_defaults(currency: str = "BTC") -> Dict[str, float]:
    """
    Get intelligent defaults for option parameters based on real market data.
    
    Args:
        currency: Currency to get defaults for
        
    Returns:
        Dictionary with smart default values
    """
    try:
        logger.info(f"🧠 Calculating smart defaults for {currency}...")
        
        # Get current price
        current_price = get_current_crypto_price(currency)
        if current_price is None:
            current_price = 30000.0  # Ultimate fallback
        
        # Get available options data
        options_data = get_available_options_data(currency)
        
        # Calculate intelligent defaults
        nearest_strike = get_nearest_strike_price(current_price, options_data, currency)
        nearest_maturity_days = get_nearest_maturity_days(options_data, currency)
        
        # Calculate intelligent volatility default
        if options_data is not None and not options_data.empty and 'implied_volatility' in options_data.columns:
            # Use median IV from recent options
            median_iv = options_data['implied_volatility'].median() * 100  # Convert to percentage
            default_volatility = max(20.0, min(200.0, median_iv))  # Clamp between 20% and 200%
        else:
            # Fallback volatility based on currency
            volatility_defaults = {
                'BTC': 80.0,   # 80% typical for BTC options
                'ETH': 90.0   # 90% typical for ETH options  
            }
            default_volatility = volatility_defaults.get(currency, 80.0)
        
        defaults = {
            'spot_price': current_price,
            'strike_price': nearest_strike,
            'time_to_expiry_days': nearest_maturity_days,
            'volatility_percent': default_volatility,
            'risk_free_rate_percent': 5.0  # Standard 5%
        }
        
        logger.info(f"✅ Smart defaults for {currency}: " +
                   f"Spot=${defaults['spot_price']:,.0f}, " +
                   f"Strike=${defaults['strike_price']:,.0f}, " +
                   f"TTM={defaults['time_to_expiry_days']:.0f}d, " +
                   f"IV={defaults['volatility_percent']:.0f}%")
        
        return defaults
        
    except Exception as e:
        logger.error(f"❌ Error calculating smart defaults: {e}")
        
        # Ultimate fallback defaults
        return {
            'spot_price': 30000.0,
            'strike_price': 32000.0,
            'time_to_expiry_days': 30.0,
            'volatility_percent': 80.0,
            'risk_free_rate_percent': 5.0
        }

# Streamlit integration helpers
def display_market_data_status(currency: str = "BTC"):
    """Display current market data status in Streamlit sidebar."""
    try:
        with st.sidebar:
            st.markdown("---")
            st.subheader("📊 Market Data")
            
            current_price = get_current_crypto_price(currency)
            if current_price:
                st.metric(f"{currency} Price", f"${current_price:,.2f}")
            
            # Show last update time
            st.caption(f"Updated: {datetime.now().strftime('%H:%M:%S')}")
            
            if st.button("🔄 Refresh Data"):
                st.cache_data.clear()
                st.experimental_rerun()
                
    except Exception as e:
        logger.warning(f"Could not display market data status: {e}")

if __name__ == "__main__":
    # Test the functions
    print("Testing market data helpers...")
    
    # Test current price
    btc_price = get_current_crypto_price("BTC")
    print(f"BTC Price: ${btc_price:,.2f}")
    
    # Test smart defaults
    defaults = get_smart_defaults("BTC")
    print(f"Smart defaults: {defaults}")
