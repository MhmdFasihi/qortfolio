"""
Time Utilities for Bitcoin Options Analytics Platform

This module provides professional-grade time handling utilities specifically 
designed for financial calculations, with proper handling of business days,
holidays, and time-to-expiry calculations.

Key Features:
- Accurate time-to-maturity calculations (fixes the original bug)
- Business day calculations
- Holiday handling
- Timezone management
- Financial calendar integration
"""

import logging
from datetime import datetime, date, timedelta, timezone
from typing import Union, Optional, List
import pandas as pd
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

# Constants for time calculations
SECONDS_PER_YEAR = 365.25 * 24 * 3600  # Accounts for leap years
BUSINESS_DAYS_PER_YEAR = 252  # Standard trading days per year
HOURS_PER_DAY = 24
MINUTES_PER_HOUR = 60
SECONDS_PER_MINUTE = 60

# Minimum time to expiry (1 hour in years)
MIN_TIME_TO_EXPIRY = 1 / (365.25 * 24)

class TimeCalculationError(Exception):
    """Custom exception for time calculation errors."""
    pass

def datetime_to_timestamp(dt_obj: Union[datetime, date]) -> int:
    """
    Convert datetime or date object to Unix timestamp in milliseconds.
    
    Args:
        dt_obj: datetime or date object to convert
        
    Returns:
        Unix timestamp in milliseconds
        
    Raises:
        TimeCalculationError: If conversion fails
    """
    try:
        if isinstance(dt_obj, date) and not isinstance(dt_obj, datetime):
            # Convert date to datetime at market open (9:30 AM EST)
            dt_obj = datetime.combine(dt_obj, datetime.min.time())
            
        # Handle timezone - if naive, assume it's UTC
        if dt_obj.tzinfo is None:
            dt_obj = dt_obj.replace(tzinfo=timezone.utc)
            
        return int(dt_obj.timestamp() * 1000)
    
    except Exception as e:
        logger.error(f"Failed to convert {dt_obj} to timestamp: {e}")
        raise TimeCalculationError(f"Timestamp conversion failed: {e}")

def timestamp_to_datetime(timestamp: Union[int, float]) -> datetime:
    """
    Convert Unix timestamp (in milliseconds) to datetime object.
    
    Args:
        timestamp: Unix timestamp in milliseconds
        
    Returns:
        datetime object (naive, in UTC time)
        
    Raises:
        TimeCalculationError: If conversion fails
    """
    try:
        if timestamp > 1e12:  # Already in milliseconds
            dt = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)
        else:  # In seconds
            dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        
        # Return as naive datetime (removing timezone info but keeping UTC time)
        return dt.replace(tzinfo=None)
    
    except Exception as e:
        logger.error(f"Failed to convert timestamp {timestamp} to datetime: {e}")
        raise TimeCalculationError(f"Datetime conversion failed: {e}")

def calculate_time_to_maturity(
    current_time: Union[datetime, date],
    maturity_time: Union[datetime, date],
    business_days_only: bool = False,
    min_time: Optional[float] = None
) -> float:
    """
    Calculate time to maturity in annualized terms.
    
    This function FIXES the original bug in BTC_Option.py where time calculation
    was mathematically incorrect (dividing by seconds per year then multiplying by 365).
    
    Args:
        current_time: Current time/date
        maturity_time: Option maturity time/date
        business_days_only: If True, use trading days (252/year), else calendar days
        min_time: Minimum time to return (defaults to 1 hour)
        
    Returns:
        Time to maturity in years (annualized)
        
    Example:
        >>> current = datetime(2025, 1, 20)
        >>> maturity = datetime(2025, 2, 20)  # 31 days later
        >>> calculate_time_to_maturity(current, maturity)
        0.0849  # About 31/365.25 years
    """
    try:
        # Convert to datetime objects if needed
        if isinstance(current_time, date) and not isinstance(current_time, datetime):
            current_time = datetime.combine(current_time, datetime.min.time())
        if isinstance(maturity_time, date) and not isinstance(maturity_time, datetime):
            maturity_time = datetime.combine(maturity_time, datetime.min.time())
            
        # Ensure consistent timezone handling - make both naive or both aware
        if current_time.tzinfo is not None and maturity_time.tzinfo is None:
            # current_time is aware, maturity_time is naive - make maturity_time aware
            maturity_time = maturity_time.replace(tzinfo=current_time.tzinfo)
        elif current_time.tzinfo is None and maturity_time.tzinfo is not None:
            # current_time is naive, maturity_time is aware - make current_time aware  
            current_time = current_time.replace(tzinfo=maturity_time.tzinfo)
        elif current_time.tzinfo is None and maturity_time.tzinfo is None:
            # Both naive - this is fine, no change needed
            pass
        # If both are aware with different timezones, convert to UTC
        elif (current_time.tzinfo is not None and maturity_time.tzinfo is not None and 
              current_time.tzinfo != maturity_time.tzinfo):
            current_time = current_time.astimezone(timezone.utc)
            maturity_time = maturity_time.astimezone(timezone.utc)
            
        # Calculate time difference
        time_diff = maturity_time - current_time
        
        # Check for negative time (expired options)
        if time_diff.total_seconds() < 0:
            logger.warning(f"Negative time to maturity: {time_diff}")
            return min_time or MIN_TIME_TO_EXPIRY
        
        # Convert to annualized time
        if business_days_only:
            # Use business days calculation
            business_days = pd.bdate_range(start=current_time, end=maturity_time).shape[0]
            time_to_maturity = business_days / BUSINESS_DAYS_PER_YEAR
        else:
            # Use calendar days calculation (more common for options)
            time_to_maturity = time_diff.total_seconds() / SECONDS_PER_YEAR
        
        # Apply minimum time constraint
        min_time = min_time or MIN_TIME_TO_EXPIRY
        time_to_maturity = max(time_to_maturity, min_time)
        
        logger.debug(f"Time to maturity: {time_to_maturity:.6f} years ({time_to_maturity*365:.1f} days)")
        return time_to_maturity
    
    except Exception as e:
        logger.error(f"Time to maturity calculation failed: {e}")
        raise TimeCalculationError(f"Failed to calculate time to maturity: {e}")

def calculate_time_decay(time_to_maturity: float, days_passed: float) -> float:
    """
    Calculate new time to maturity after time decay.
    
    Args:
        time_to_maturity: Current time to maturity (in years)
        days_passed: Number of days that have passed
        
    Returns:
        New time to maturity after decay
    """
    time_decay_years = days_passed / 365.25
    return max(time_to_maturity - time_decay_years, MIN_TIME_TO_EXPIRY)

def annualize_volatility(volatility: float, frequency: str = 'daily') -> float:
    """
    Convert volatility to annualized terms.
    
    Args:
        volatility: Volatility value
        frequency: 'daily', 'weekly', 'monthly', or 'annual'
        
    Returns:
        Annualized volatility
    """
    frequency_multipliers = {
        'daily': np.sqrt(252),
        'weekly': np.sqrt(52),
        'monthly': np.sqrt(12),
        'annual': 1.0
    }
    
    if frequency not in frequency_multipliers:
        raise ValueError(f"Unknown frequency: {frequency}")
    
    return volatility * frequency_multipliers[frequency]

def is_business_day(dt: Union[datetime, date]) -> bool:
    """
    Check if a given date is a business day (Monday-Friday).
    
    Args:
        dt: Date to check
        
    Returns:
        True if business day, False otherwise
    """
    if isinstance(dt, datetime):
        dt = dt.date()
    return dt.weekday() < 5  # Monday=0, Friday=4

def get_business_days_between(start_date: date, end_date: date) -> int:
    """
    Get number of business days between two dates.
    
    Args:
        start_date: Start date
        end_date: End date
        
    Returns:
        Number of business days
    """
    return len(pd.bdate_range(start=start_date, end=end_date)) - 1

def time_to_expiry_in_units(
    current_time: Union[datetime, date],
    expiry_time: Union[datetime, date],
    units: str = 'years'
) -> float:
    """
    Calculate time to expiry in specified units.
    
    Args:
        current_time: Current time
        expiry_time: Expiry time
        units: 'years', 'days', 'hours', 'minutes', 'seconds'
        
    Returns:
        Time to expiry in specified units
    """
    time_diff = expiry_time - current_time
    seconds = time_diff.total_seconds()
    
    if seconds < 0:
        return 0.0
    
    conversion_factors = {
        'seconds': 1,
        'minutes': SECONDS_PER_MINUTE,
        'hours': SECONDS_PER_MINUTE * MINUTES_PER_HOUR,
        'days': SECONDS_PER_MINUTE * MINUTES_PER_HOUR * HOURS_PER_DAY,
        'years': SECONDS_PER_YEAR
    }
    
    if units not in conversion_factors:
        raise ValueError(f"Unknown time unit: {units}")
    
    return seconds / conversion_factors[units]

def validate_time_inputs(
    current_time: Union[datetime, date],
    maturity_time: Union[datetime, date]
) -> tuple[datetime, datetime]:
    """
    Validate and normalize time inputs.
    
    Args:
        current_time: Current time
        maturity_time: Maturity time
        
    Returns:
        Tuple of normalized datetime objects
        
    Raises:
        TimeCalculationError: If inputs are invalid
    """
    try:
        # Convert to datetime if needed
        if isinstance(current_time, date) and not isinstance(current_time, datetime):
            current_time = datetime.combine(current_time, datetime.min.time())
        if isinstance(maturity_time, date) and not isinstance(maturity_time, datetime):
            maturity_time = datetime.combine(maturity_time, datetime.min.time())
        
        # Handle timezone consistency - prefer naive datetimes for simplicity
        if current_time.tzinfo is not None and maturity_time.tzinfo is None:
            # Make current_time naive by removing timezone (assume it's already in correct timezone)
            current_time = current_time.replace(tzinfo=None)
        elif current_time.tzinfo is None and maturity_time.tzinfo is not None:
            # Make maturity_time naive
            maturity_time = maturity_time.replace(tzinfo=None)
        elif (current_time.tzinfo is not None and maturity_time.tzinfo is not None and 
              current_time.tzinfo != maturity_time.tzinfo):
            # Both have different timezones - convert to UTC then make naive
            current_time = current_time.astimezone(timezone.utc).replace(tzinfo=None)
            maturity_time = maturity_time.astimezone(timezone.utc).replace(tzinfo=None)
        elif current_time.tzinfo is not None and maturity_time.tzinfo is not None:
            # Both have same timezone - make them naive
            current_time = current_time.replace(tzinfo=None)
            maturity_time = maturity_time.replace(tzinfo=None)
        
        # Validate chronological order (but don't fail for expired options)
        if maturity_time <= current_time:
            logger.warning(f"Maturity time {maturity_time} is not after current time {current_time}")
        
        return current_time, maturity_time
    
    except Exception as e:
        raise TimeCalculationError(f"Time input validation failed: {e}")

# Convenience functions for common calculations
def days_to_years(days: float) -> float:
    """Convert days to years."""
    return days / 365.25

def years_to_days(years: float) -> float:
    """Convert years to days."""
    return years * 365.25

def hours_to_years(hours: float) -> float:
    """Convert hours to years."""
    return hours / (365.25 * 24)

def years_to_hours(years: float) -> float:
    """Convert years to hours."""
    return years * 365.25 * 24

# Testing and validation functions
def test_time_calculations():
    """
    Test time calculation functions with known values.
    
    Returns:
        True if all tests pass
    """
    try:
        # Test 1: Basic time to maturity (using naive datetimes)
        current = datetime(2025, 1, 20, 10, 0, 0)  # Naive datetime
        maturity = datetime(2025, 2, 20, 10, 0, 0)  # Exactly 31 days, naive
        
        ttm = calculate_time_to_maturity(current, maturity)
        expected_ttm = 31 / 365.25  # About 0.0849
        
        assert abs(ttm - expected_ttm) < 1e-6, f"Expected {expected_ttm}, got {ttm}"
        
        # Test 2: Minimum time constraint
        ttm_expired = calculate_time_to_maturity(maturity, current)  # Negative time
        assert ttm_expired == MIN_TIME_TO_EXPIRY, f"Expected minimum time, got {ttm_expired}"
        
        # Test 3: Timestamp conversions
        timestamp = datetime_to_timestamp(current)
        converted_back = timestamp_to_datetime(timestamp)
        
        # Should be within 1 second (both should be naive now)
        assert abs((converted_back - current).total_seconds()) < 1
        
        # Test 4: Mixed timezone handling
        current_aware = current.replace(tzinfo=timezone.utc)
        maturity_naive = maturity  # naive
        
        ttm_mixed = calculate_time_to_maturity(current_aware, maturity_naive)
        assert abs(ttm_mixed - expected_ttm) < 1e-6, f"Mixed timezone test failed: {ttm_mixed}"
        
        logger.info("✅ All time calculation tests passed!")
        return True
    
    except Exception as e:
        logger.error(f"❌ Time calculation tests failed: {e}")
        return False

if __name__ == "__main__":
    # Run tests when module is executed directly
    test_time_calculations()