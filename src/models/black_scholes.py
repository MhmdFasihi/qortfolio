"""
# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com
Professional Black-Scholes Options Pricing Model

This module provides a comprehensive Black-Scholes implementation with complete
Greeks calculations, extracted and enhanced from options_claude.py V2.

Key Features:
- Accurate Black-Scholes pricing for European options
- Complete Greeks: Delta, Gamma, Theta, Vega, Rho
- Fixed time-to-maturity calculations using our time utilities
- Vectorized operations for performance
- Professional error handling and validation
- Model validation against known benchmarks
"""

import logging
import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, Union, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

# Import our fixed time utilities
from ..utils.time_utils import calculate_time_to_maturity, TimeCalculationError

# Configure logging
logger = logging.getLogger(__name__)

class OptionType(Enum):
    """Option types enumeration."""
    CALL = "call"
    PUT = "put"

@dataclass
class OptionParameters:
    """
    Data class to hold option parameters for pricing.
    """
    spot_price: float           # Current underlying price (S)
    strike_price: float         # Strike price (K)
    time_to_expiry: float      # Time to expiry in years (T)
    volatility: float          # Implied volatility (σ)
    risk_free_rate: float      # Risk-free interest rate (r)
    dividend_yield: float = 0.0 # Dividend yield (q)
    option_type: OptionType = OptionType.CALL

    def __post_init__(self):
        """Validate parameters after initialization."""
        self._validate_parameters()

    def _validate_parameters(self):
        """Validate all option parameters."""
        if self.spot_price <= 0:
            raise ValueError(f"Spot price must be positive: {self.spot_price}")
        if self.strike_price <= 0:
            raise ValueError(f"Strike price must be positive: {self.strike_price}")
        if self.time_to_expiry <= 0:
            raise ValueError(f"Time to expiry must be positive: {self.time_to_expiry}")
        if self.volatility < 0:
            raise ValueError(f"Volatility cannot be negative: {self.volatility}")
        if self.volatility > 10:  # 1000% volatility cap
            logger.warning(f"Very high volatility: {self.volatility:.1%}")

@dataclass
class GreeksResult:
    """
    Data class to hold calculated Greeks.
    """
    delta: float    # Price sensitivity to underlying
    gamma: float    # Delta sensitivity to underlying  
    theta: float    # Price sensitivity to time decay
    vega: float     # Price sensitivity to volatility
    rho: float      # Price sensitivity to interest rate
    option_price: float = 0.0

class BlackScholesError(Exception):
    """Custom exception for Black-Scholes calculation errors."""
    pass

class BlackScholesModel:
    """
    Professional Black-Scholes options pricing model.
    
    This class provides comprehensive Black-Scholes pricing and Greeks
    calculations, enhanced from the options_claude.py V2 implementation
    with fixed time calculations and professional error handling.
    """
    
    def __init__(self, default_risk_free_rate: float = 0.05):
        """
        Initialize Black-Scholes model.
        
        Args:
            default_risk_free_rate: Default risk-free rate (5% annual)
        """
        self.default_risk_free_rate = default_risk_free_rate
        self.calculations_performed = 0
        
        logger.info(f"BlackScholesModel initialized with default rate: {default_risk_free_rate:.2%}")

    def _calculate_d1_d2(self, params: OptionParameters) -> Tuple[float, float]:
        """
        Calculate d1 and d2 parameters for Black-Scholes formula with improved numerical stability.
        
        Args:
            params: Option parameters
            
        Returns:
            Tuple of (d1, d2)
        """
        try:
            S = params.spot_price
            K = params.strike_price
            T = params.time_to_expiry
            sigma = params.volatility
            r = params.risk_free_rate
            q = params.dividend_yield

            # IMPROVED: Handle edge cases better
            if T <= 0:
                raise BlackScholesError("Time to expiry must be positive")
            if sigma <= 0:
                raise BlackScholesError("Volatility must be positive")
            if S <= 0 or K <= 0:
                raise BlackScholesError("Prices must be positive")
            
            # Calculate sigma * sqrt(T) once for efficiency and stability
            sigma_sqrt_t = sigma * np.sqrt(T)
            
            # Check for very small sigma*sqrt(T) which could cause numerical issues
            if sigma_sqrt_t < 1e-8:
                logger.warning(f"Very small sigma*sqrt(T): {sigma_sqrt_t}, using minimum value")
                sigma_sqrt_t = 1e-8
            
            # Standard Black-Scholes d1 calculation with improved numerical stability
            log_s_k = np.log(S / K)
            drift_term = (r - q + 0.5 * sigma**2) * T
            
            d1 = (log_s_k + drift_term) / sigma_sqrt_t
            d2 = d1 - sigma_sqrt_t
            
            # ADDED: Check for extreme values that might cause issues
            if abs(d1) > 10 or abs(d2) > 10:
                logger.warning(f"Extreme d1/d2 values: d1={d1:.4f}, d2={d2:.4f}")
            
            return d1, d2
        
        except Exception as e:
            raise BlackScholesError(f"Failed to calculate d1, d2: {e}")

    def option_price(self, params: OptionParameters) -> float:
        """
        Calculate Black-Scholes option price.
        
        Args:
            params: Option parameters
            
        Returns:
            Option price
        """
        try:
            S = params.spot_price
            K = params.strike_price
            T = params.time_to_expiry
            r = params.risk_free_rate
            q = params.dividend_yield
            
            d1, d2 = self._calculate_d1_d2(params)
            
            if params.option_type == OptionType.CALL:
                # Call option price
                price = (S * np.exp(-q * T) * norm.cdf(d1) - 
                        K * np.exp(-r * T) * norm.cdf(d2))
            else:
                # Put option price  
                price = (K * np.exp(-r * T) * norm.cdf(-d2) - 
                        S * np.exp(-q * T) * norm.cdf(-d1))
            
            self.calculations_performed += 1
            return max(price, 0.0)  # Ensure non-negative price
        
        except Exception as e:
            raise BlackScholesError(f"Option price calculation failed: {e}")

    def calculate_greeks(self, params: OptionParameters) -> GreeksResult:
        """
        Calculate all Greeks for an option.
        
        This is the enhanced version of _calculate_greeks from options_claude.py V2,
        with fixed time calculations and improved error handling.
        
        Args:
            params: Option parameters
            
        Returns:
            GreeksResult with all calculated Greeks
        """
        try:
            S = params.spot_price
            K = params.strike_price
            T = params.time_to_expiry
            sigma = params.volatility
            r = params.risk_free_rate
            q = params.dividend_yield
            is_call = (params.option_type == OptionType.CALL)

            d1, d2 = self._calculate_d1_d2(params)
            
            # Common terms
            sqrt_T = np.sqrt(T)
            exp_neg_qT = np.exp(-q * T)
            exp_neg_rT = np.exp(-r * T)
            norm_pdf_d1 = norm.pdf(d1)
            
            # Delta calculation
            if is_call:
                delta = exp_neg_qT * norm.cdf(d1)
            else:
                delta = -exp_neg_qT * norm.cdf(-d1)

            # Gamma calculation (same for calls and puts)
            gamma = (exp_neg_qT * norm_pdf_d1) / (S * sigma * sqrt_T)
            
            # Vega calculation (same for calls and puts, per 1% change in volatility)
            vega = S * exp_neg_qT * sqrt_T * norm_pdf_d1 * 0.01  # For 1% vol change
            
            # Theta calculation (per day)
            theta_part1 = -(S * exp_neg_qT * norm_pdf_d1 * sigma) / (2 * sqrt_T)
            
            if is_call:
                theta_part2 = -r * K * exp_neg_rT * norm.cdf(d2)
                theta_part3 = q * S * exp_neg_qT * norm.cdf(d1)
                theta = theta_part1 + theta_part2 + theta_part3
            else:
                theta_part2 = r * K * exp_neg_rT * norm.cdf(-d2)
                theta_part3 = -q * S * exp_neg_qT * norm.cdf(-d1)
                theta = theta_part1 + theta_part2 + theta_part3
            
            # Convert theta to daily (divide by 365)
            theta = theta / 365.0
            
            # Rho calculation (per 1% change in interest rate)
            if is_call:
                rho = K * T * exp_neg_rT * norm.cdf(d2) * 0.01
            else:
                rho = -K * T * exp_neg_rT * norm.cdf(-d2) * 0.01
            
            # Calculate option price
            option_price = self.option_price(params)
            
            return GreeksResult(
                delta=delta,
                gamma=gamma,
                theta=theta,
                vega=vega,
                rho=rho,
                option_price=option_price
            )
        
        except Exception as e:
            raise BlackScholesError(f"Greeks calculation failed: {e}")

    def calculate_greeks_for_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Greeks for a DataFrame of options (vectorized version of V2's apply method).
        
        Expected DataFrame columns:
        - index_price (or underlying_price): Current underlying price
        - strike_price: Strike price  
        - time_to_maturity: Time to expiry in years
        - implied_volatility (or iv): Implied volatility
        - option_type: 'call' or 'put'
        
        Args:
            df: DataFrame with option data
            
        Returns:
            DataFrame with added Greeks columns
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for Greeks calculation")
            return df
        
        logger.info(f"Calculating Greeks for {len(df)} options...")
        
        # Standardize column names
        df_copy = df.copy()
        
        # Map different possible column names
        column_mapping = {
            'index_price': 'spot_price',
            'underlying_price': 'spot_price', 
            'iv': 'implied_volatility',
            'time_to_maturity': 'time_to_expiry'
        }
        
        for old_name, new_name in column_mapping.items():
            if old_name in df_copy.columns and new_name not in df_copy.columns:
                df_copy[new_name] = df_copy[old_name]
        
        # Validate required columns
        required_columns = ['spot_price', 'strike_price', 'time_to_expiry', 'implied_volatility', 'option_type']
        missing_columns = [col for col in required_columns if col not in df_copy.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # FIXED: Improved time handling with better edge cases
        def calculate_row_greeks(row):
            try:
                # Create option parameters with improved time handling
                option_type = OptionType.CALL if str(row['option_type']).lower() in ['call', 'c'] else OptionType.PUT
                
                # IMPROVED: Better minimum time handling
                raw_time = float(row['time_to_expiry'])
                
                # Handle different time edge cases
                if raw_time <= 0:
                    # Expired options - use very small time for numerical stability
                    time_to_expiry = 1 / (365.25 * 24 * 60)  # 1 minute in years
                    logger.debug(f"Expired option detected, using minimum time: {time_to_expiry}")
                elif raw_time < 1 / (365.25 * 24):  # Less than 1 day
                    # Very short-term options - use at least 1 hour
                    time_to_expiry = max(raw_time, 1 / (365.25 * 24))
                    if time_to_expiry != raw_time:
                        logger.debug(f"Very short time adjusted: {raw_time} -> {time_to_expiry}")
                else:
                    # Normal case
                    time_to_expiry = raw_time
                
                # IMPROVED: Better volatility handling
                volatility = float(row['implied_volatility'])
                if volatility <= 0:
                    logger.warning(f"Non-positive volatility {volatility}, using minimum 1%")
                    volatility = 0.01  # 1% minimum
                elif volatility > 10:  # 1000%
                    logger.warning(f"Extremely high volatility {volatility}, capping at 1000%")
                    volatility = 10.0
                
                # IMPROVED: Better price handling
                spot_price = float(row['spot_price'])
                strike_price = float(row['strike_price'])
                
                if spot_price <= 0:
                    raise ValueError(f"Invalid spot price: {spot_price}")
                if strike_price <= 0:
                    raise ValueError(f"Invalid strike price: {strike_price}")
                
                params = OptionParameters(
                    spot_price=spot_price,
                    strike_price=strike_price,
                    time_to_expiry=time_to_expiry,
                    volatility=volatility,
                    risk_free_rate=self.default_risk_free_rate,
                    option_type=option_type
                )
                
                greeks = self.calculate_greeks(params)
                
                return pd.Series({
                    'delta': greeks.delta,
                    'gamma': greeks.gamma,
                    'theta': greeks.theta,
                    'vega': greeks.vega,
                    'rho': greeks.rho,
                    'bs_price': greeks.option_price
                })
                
            except Exception as e:
                logger.warning(f"Greeks calculation failed for row: {e}")
                return pd.Series({
                    'delta': np.nan,
                    'gamma': np.nan,
                    'theta': np.nan,
                    'vega': np.nan,
                    'rho': np.nan,
                    'bs_price': np.nan
                })
        
        # Calculate Greeks for all rows
        greeks_df = df_copy.apply(calculate_row_greeks, axis=1)
        
        # Combine with original data
        result_df = pd.concat([df, greeks_df], axis=1)
        
        # ADDED: Data quality checks and reporting
        valid_calculations = greeks_df['delta'].notna().sum()
        invalid_calculations = len(df) - valid_calculations
        
        # Report any issues
        if invalid_calculations > 0:
            logger.warning(f"{invalid_calculations} options had calculation issues")
            
            # Check for common issues
            zero_time_count = (df_copy['time_to_expiry'] <= 0).sum()
            if zero_time_count > 0:
                logger.warning(f"{zero_time_count} options had zero/negative time to expiry")
            
            zero_vol_count = (df_copy['implied_volatility'] <= 0).sum()
            if zero_vol_count > 0:
                logger.warning(f"{zero_vol_count} options had zero/negative volatility")
        
        # Log success statistics
        logger.info(f"✅ Successfully calculated Greeks for {valid_calculations}/{len(df)} options")
        if valid_calculations > 0:
            # Log some statistics about the Greeks
            mean_delta = result_df['delta'].mean()
            mean_gamma = result_df['gamma'].mean()
            logger.info(f"   Average Delta: {mean_delta:.4f}, Average Gamma: {mean_gamma:.6f}")
        
        return result_df
        
        # Apply Greeks calculation to each row
        def calculate_row_greeks(row):
            try:
                # Create option parameters
                option_type = OptionType.CALL if str(row['option_type']).lower() in ['call', 'c'] else OptionType.PUT
                
                params = OptionParameters(
                    spot_price=float(row['spot_price']),
                    strike_price=float(row['strike_price']),
                    time_to_expiry=max(float(row['time_to_expiry']), 1e-6),  # Minimum time
                    volatility=float(row['implied_volatility']),
                    risk_free_rate=self.default_risk_free_rate,
                    option_type=option_type
                )
                
                greeks = self.calculate_greeks(params)
                
                return pd.Series({
                    'delta': greeks.delta,
                    'gamma': greeks.gamma,
                    'theta': greeks.theta,
                    'vega': greeks.vega,
                    'rho': greeks.rho,
                    'bs_price': greeks.option_price
                })
                
            except Exception as e:
                logger.warning(f"Greeks calculation failed for row: {e}")
                return pd.Series({
                    'delta': np.nan,
                    'gamma': np.nan,
                    'theta': np.nan,
                    'vega': np.nan,
                    'rho': np.nan,
                    'bs_price': np.nan
                })
        
        # Calculate Greeks for all rows
        greeks_df = df_copy.apply(calculate_row_greeks, axis=1)
        
        # Combine with original data
        result_df = pd.concat([df, greeks_df], axis=1)
        
        # Log statistics
        valid_calculations = greeks_df['delta'].notna().sum()
        logger.info(f"✅ Successfully calculated Greeks for {valid_calculations}/{len(df)} options")
        
        return result_df

    def implied_volatility(self, market_price: float, params: OptionParameters, 
                          max_iterations: int = 100, tolerance: float = 1e-6) -> float:
        """
        Calculate implied volatility using Newton-Raphson method.
        
        Args:
            market_price: Observed market price
            params: Option parameters (volatility will be solved for)
            max_iterations: Maximum iterations for convergence
            tolerance: Convergence tolerance
            
        Returns:
            Implied volatility
        """
        try:
            # Initial guess
            iv = 0.2  # 20% initial guess
            
            for i in range(max_iterations):
                # Create parameters with current IV guess
                test_params = OptionParameters(
                    spot_price=params.spot_price,
                    strike_price=params.strike_price,
                    time_to_expiry=params.time_to_expiry,
                    volatility=iv,
                    risk_free_rate=params.risk_free_rate,
                    dividend_yield=params.dividend_yield,
                    option_type=params.option_type
                )
                
                # Calculate theoretical price and vega
                theoretical_price = self.option_price(test_params)
                greeks = self.calculate_greeks(test_params)
                
                # Price difference
                price_diff = theoretical_price - market_price
                
                # Check convergence
                if abs(price_diff) < tolerance:
                    return iv
                
                # Newton-Raphson update
                if greeks.vega > 1e-6:  # Avoid division by zero
                    iv = iv - price_diff / (greeks.vega * 100)  # Convert vega back to per unit
                    iv = max(0.001, min(10.0, iv))  # Keep IV in reasonable bounds
                else:
                    break
            
            logger.warning(f"Implied volatility did not converge after {max_iterations} iterations")
            return iv
        
        except Exception as e:
            raise BlackScholesError(f"Implied volatility calculation failed: {e}")

    def validate_against_benchmarks(self) -> bool:
        """
        Validate the model against known option pricing benchmarks.
        
        Returns:
            True if all benchmarks pass
        """
        try:
            logger.info("Running Black-Scholes model validation...")
            
            # Test Case 1: At-the-money call option
            params1 = OptionParameters(
                spot_price=100.0,
                strike_price=100.0,
                time_to_expiry=1.0,
                volatility=0.20,
                risk_free_rate=0.05,
                option_type=OptionType.CALL
            )
            
            price1 = self.option_price(params1)
            greeks1 = self.calculate_greeks(params1)
            
            # Expected values (approximate)
            expected_price1 = 10.45  # Theoretical BS price
            expected_delta1 = 0.637  # Theoretical delta
            
            if abs(price1 - expected_price1) > 0.1:
                logger.error(f"Price validation failed: {price1} vs expected {expected_price1}")
                return False
            
            if abs(greeks1.delta - expected_delta1) > 0.01:
                logger.error(f"Delta validation failed: {greeks1.delta} vs expected {expected_delta1}")
                return False
            
            # Test Case 2: Deep in-the-money put
            params2 = OptionParameters(
                spot_price=80.0,
                strike_price=100.0,
                time_to_expiry=0.25,
                volatility=0.30,
                risk_free_rate=0.05,
                option_type=OptionType.PUT
            )
            
            price2 = self.option_price(params2)
            greeks2 = self.calculate_greeks(params2)
            
            # Put should have negative delta
            if greeks2.delta >= 0:
                logger.error(f"Put delta should be negative: {greeks2.delta}")
                return False
            
            # Price should be reasonable
            if price2 < 15 or price2 > 25:
                logger.error(f"Put price seems unreasonable: {price2}")
                return False
            
            logger.info("✅ All validation benchmarks passed!")
            return True
        
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False

# Convenience functions for quick calculations
def calculate_option_greeks(spot_price: float, strike_price: float, time_to_expiry: float,
                           volatility: float, risk_free_rate: float = 0.05,
                           option_type: str = "call") -> Dict[str, float]:
    """
    Convenience function to calculate Greeks for a single option.
    
    Args:
        spot_price: Current underlying price
        strike_price: Strike price
        time_to_expiry: Time to expiry in years
        volatility: Implied volatility (decimal, e.g., 0.20 for 20%)
        risk_free_rate: Risk-free rate (default: 5%)
        option_type: 'call' or 'put'
        
    Returns:
        Dictionary with Greeks and option price
    """
    model = BlackScholesModel(risk_free_rate)
    
    params = OptionParameters(
        spot_price=spot_price,
        strike_price=strike_price,
        time_to_expiry=time_to_expiry,
        volatility=volatility,
        risk_free_rate=risk_free_rate,
        option_type=OptionType.CALL if option_type.lower() == 'call' else OptionType.PUT
    )
    
    greeks = model.calculate_greeks(params)
    
    return {
        'option_price': greeks.option_price,
        'delta': greeks.delta,
        'gamma': greeks.gamma,
        'theta': greeks.theta,
        'vega': greeks.vega,
        'rho': greeks.rho
    }

# Testing and validation
def test_black_scholes_model():
    """
    Test Black-Scholes model functionality.
    
    Returns:
        True if all tests pass
    """
    try:
        logger.info("Testing Black-Scholes model...")
        
        # Initialize model
        model = BlackScholesModel()
        
        # Run validation
        if not model.validate_against_benchmarks():
            return False
        
        # Test DataFrame functionality
        test_data = pd.DataFrame({
            'spot_price': [100, 100, 100],
            'strike_price': [95, 100, 105],
            'time_to_expiry': [0.25, 0.25, 0.25],
            'implied_volatility': [0.20, 0.20, 0.20],
            'option_type': ['call', 'call', 'call']
        })
        
        result_df = model.calculate_greeks_for_dataframe(test_data)
        
        # Check that Greeks were calculated
        if 'delta' not in result_df.columns:
            logger.error("Greeks not calculated for DataFrame")
            return False
        
        # Check that deltas are reasonable for calls
        if not all(result_df['delta'] > 0):
            logger.error("Call deltas should be positive")
            return False
        
        logger.info("✅ All Black-Scholes tests passed!")
        return True
    
    except Exception as e:
        logger.error(f"Black-Scholes testing failed: {e}")
        return False

if __name__ == "__main__":
    # Run tests when module is executed directly
    test_black_scholes_model()
