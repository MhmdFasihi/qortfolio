
"""
# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com
Taylor Expansion PnL Simulator for Bitcoin Options Analytics Platform

This module implements the core requested feature: Taylor expansion PnL analysis
using the formula: ΔC ≈ δΔS + ½γ(ΔS)² + θΔt + νΔσ

Key Features:
- Professional Taylor expansion PnL calculation
- Multi-scenario stress testing
- Integration with Black-Scholes Greeks
- Portfolio-level analysis
- Risk metrics (VaR, CVaR)
- Comprehensive scenario analysis
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import itertools

# Import our models and utilities
from ..models.black_scholes import BlackScholesModel, OptionParameters, GreeksResult, OptionType
from ..utils.time_utils import calculate_time_decay

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class ScenarioParameters:
    """Parameters for scenario analysis."""
    spot_shocks: List[float] = field(default_factory=lambda: [-0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2])
    vol_shocks: List[float] = field(default_factory=lambda: [-0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5])
    time_decay_days: List[float] = field(default_factory=lambda: [0, 1, 7, 30])
    
    def __post_init__(self):
        """Validate scenario parameters."""
        if not self.spot_shocks:
            raise ValueError("Spot shocks cannot be empty")
        if not self.vol_shocks:
            raise ValueError("Vol shocks cannot be empty")
        if not self.time_decay_days:
            raise ValueError("Time decay days cannot be empty")

@dataclass 
class PnLComponents:
    """Breakdown of PnL components from Taylor expansion."""
    delta_pnl: float        # δΔS component
    gamma_pnl: float        # ½γ(ΔS)² component  
    theta_pnl: float        # θΔt component
    vega_pnl: float         # νΔσ component
    total_pnl: float        # Sum of all components
    
    # Scenario details
    spot_shock: float = 0.0
    vol_shock: float = 0.0  
    time_decay: float = 0.0
    
    # Original option details
    original_price: float = 0.0
    new_theoretical_price: float = 0.0

@dataclass
class ScenarioResult:
    """Result of a single scenario analysis."""
    scenario_id: str
    pnl_components: PnLComponents
    option_parameters: OptionParameters
    original_greeks: GreeksResult
    
    @property
    def pnl_percentage(self) -> float:
        """PnL as percentage of original option price."""
        if self.pnl_components.original_price > 0:
            return (self.pnl_components.total_pnl / self.pnl_components.original_price) * 100
        return 0.0

class TaylorExpansionPnLError(Exception):
    """Custom exception for PnL calculation errors."""
    pass

class TaylorExpansionPnL:
    """
    Professional Taylor Expansion PnL Simulator.
    
    This class implements the core requested feature using the formula:
    ΔC ≈ δΔS + ½γ(ΔS)² + θΔt + νΔσ
    
    Where:
    - δ (delta): Price sensitivity to underlying
    - γ (gamma): Delta sensitivity to underlying  
    - θ (theta): Price sensitivity to time decay
    - ν (vega): Price sensitivity to volatility
    """
    
    def __init__(self, black_scholes_model: Optional[BlackScholesModel] = None):
        """
        Initialize Taylor expansion PnL simulator.
        
        Args:
            black_scholes_model: Black-Scholes model instance (creates default if None)
        """
        self.bs_model = black_scholes_model or BlackScholesModel()
        self.calculations_performed = 0
        
        logger.info("TaylorExpansionPnL initialized with Black-Scholes integration")

    def calculate_pnl_components(self,
                                option_params: OptionParameters,
                                spot_shock: float = 0.0,
                                vol_shock: float = 0.0,
                                time_decay_days: float = 0.0) -> PnLComponents:
        """
        Calculate PnL using Taylor expansion formula: ΔC ≈ δΔS + ½γ(ΔS)² + θΔt + νΔσ
        
        Args:
            option_params: Original option parameters
            spot_shock: Relative change in spot price (e.g., 0.1 for +10%)
            vol_shock: Relative change in volatility (e.g., 0.2 for +20%)
            time_decay_days: Number of days of time decay
            
        Returns:
            PnLComponents with detailed breakdown
        """
        try:
            # Calculate original Greeks
            original_greeks = self.bs_model.calculate_greeks(option_params)
            original_price = original_greeks.option_price
            
            # Calculate changes
            spot_change = option_params.spot_price * spot_shock
            vol_change = vol_shock  # Volatility shock is absolute
            time_change = time_decay_days / 365.25  # Convert days to years
            
            # Taylor expansion components
            # 1. Delta component: δΔS
            delta_pnl = original_greeks.delta * spot_change
            
            # 2. Gamma component: ½γ(ΔS)²
            gamma_pnl = 0.5 * original_greeks.gamma * (spot_change ** 2)
            
            # 3. Theta component: θΔt
            # Note: theta is already daily, so multiply by days directly
            theta_pnl = original_greeks.theta * time_decay_days
            
            # 4. Vega component: νΔσ
            # Note: vega is per 1% vol change, convert vol_shock to percentage points
            vega_pnl = original_greeks.vega * (vol_shock * 100)
            
            # Total PnL
            total_pnl = delta_pnl + gamma_pnl + theta_pnl + vega_pnl
            
            # Calculate new theoretical price for validation
            new_params = OptionParameters(
                spot_price=max(option_params.spot_price + spot_change, 0.01),
                strike_price=option_params.strike_price,
                time_to_expiry=max(option_params.time_to_expiry - time_change, 1e-6),
                volatility=max(option_params.volatility + vol_change, 0.001),
                risk_free_rate=option_params.risk_free_rate,
                dividend_yield=option_params.dividend_yield,
                option_type=option_params.option_type
            )
            
            new_theoretical_price = self.bs_model.option_price(new_params)
            
            self.calculations_performed += 1
            
            return PnLComponents(
                delta_pnl=delta_pnl,
                gamma_pnl=gamma_pnl,
                theta_pnl=theta_pnl,
                vega_pnl=vega_pnl,
                total_pnl=total_pnl,
                spot_shock=spot_shock,
                vol_shock=vol_shock,
                time_decay=time_decay_days,
                original_price=original_price,
                new_theoretical_price=new_theoretical_price
            )
            
        except Exception as e:
            logger.error(f"PnL calculation failed: {e}")
            raise TaylorExpansionPnLError(f"Failed to calculate PnL components: {e}")

    def analyze_single_scenario(self,
                               option_params: OptionParameters,
                               spot_shock: float = 0.0,
                               vol_shock: float = 0.0,
                               time_decay_days: float = 0.0,
                               scenario_id: Optional[str] = None) -> ScenarioResult:
        """
        Analyze a single scenario with detailed results.
        
        Args:
            option_params: Option parameters
            spot_shock: Spot price shock
            vol_shock: Volatility shock  
            time_decay_days: Time decay in days
            scenario_id: Optional scenario identifier
            
        Returns:
            ScenarioResult with comprehensive analysis
        """
        if scenario_id is None:
            scenario_id = f"S{spot_shock:+.1%}_V{vol_shock:+.1%}_T{time_decay_days:.0f}d"
        
        # Calculate PnL components
        pnl_components = self.calculate_pnl_components(
            option_params, spot_shock, vol_shock, time_decay_days
        )
        
        # Get original Greeks
        original_greeks = self.bs_model.calculate_greeks(option_params)
        
        return ScenarioResult(
            scenario_id=scenario_id,
            pnl_components=pnl_components,
            option_parameters=option_params,
            original_greeks=original_greeks
        )

    def analyze_scenarios(self,
                         option_params: OptionParameters,
                         scenario_params: Optional[ScenarioParameters] = None) -> List[ScenarioResult]:
        """
        Analyze multiple scenarios using Taylor expansion.
        
        Args:
            option_params: Base option parameters
            scenario_params: Scenario analysis parameters
            
        Returns:
            List of ScenarioResult objects
        """
        if scenario_params is None:
            scenario_params = ScenarioParameters()
        
        logger.info(f"Analyzing {len(scenario_params.spot_shocks) * len(scenario_params.vol_shocks) * len(scenario_params.time_decay_days)} scenarios...")
        
        results = []
        
        # Generate all combinations of scenarios
        for spot_shock, vol_shock, time_decay in itertools.product(
            scenario_params.spot_shocks,
            scenario_params.vol_shocks, 
            scenario_params.time_decay_days
        ):
            try:
                result = self.analyze_single_scenario(
                    option_params, spot_shock, vol_shock, time_decay
                )
                results.append(result)
                
            except Exception as e:
                logger.warning(f"Scenario failed (S{spot_shock:+.1%}, V{vol_shock:+.1%}, T{time_decay:.0f}d): {e}")
        
        logger.info(f"Successfully analyzed {len(results)} scenarios")
        return results

    def calculate_risk_metrics(self, scenario_results: List[ScenarioResult]) -> Dict[str, float]:
        """
        Calculate risk metrics from scenario results.
        
        Args:
            scenario_results: List of scenario results
            
        Returns:
            Dictionary with risk metrics
        """
        if not scenario_results:
            return {}
        
        # Extract PnL values
        pnl_values = [result.pnl_components.total_pnl for result in scenario_results]
        pnl_percentages = [result.pnl_percentage for result in scenario_results]
        
        # Basic statistics
        metrics = {
            'mean_pnl': np.mean(pnl_values),
            'std_pnl': np.std(pnl_values),
            'min_pnl': np.min(pnl_values),
            'max_pnl': np.max(pnl_values),
            'median_pnl': np.median(pnl_values),
            'mean_pnl_pct': np.mean(pnl_percentages),
            'std_pnl_pct': np.std(pnl_percentages)
        }
        
        # Value at Risk (VaR) - 95% and 99% confidence levels
        metrics['var_95_pnl'] = np.percentile(pnl_values, 5)  # 5th percentile
        metrics['var_99_pnl'] = np.percentile(pnl_values, 1)  # 1st percentile
        metrics['var_95_pct'] = np.percentile(pnl_percentages, 5)
        metrics['var_99_pct'] = np.percentile(pnl_percentages, 1)
        
        # Conditional Value at Risk (CVaR) - Expected Shortfall
        var_95_threshold = metrics['var_95_pnl']
        var_99_threshold = metrics['var_99_pnl']
        
        tail_95 = [pnl for pnl in pnl_values if pnl <= var_95_threshold]
        tail_99 = [pnl for pnl in pnl_values if pnl <= var_99_threshold]
        
        if tail_95:
            metrics['cvar_95_pnl'] = np.mean(tail_95)
        if tail_99:
            metrics['cvar_99_pnl'] = np.mean(tail_99)
        
        # Probability of loss
        losses = [pnl for pnl in pnl_values if pnl < 0]
        metrics['prob_loss'] = len(losses) / len(pnl_values) * 100
        
        # Expected gain/loss
        gains = [pnl for pnl in pnl_values if pnl > 0]
        if gains:
            metrics['expected_gain'] = np.mean(gains)
        if losses:
            metrics['expected_loss'] = np.mean(losses)
        
        return metrics

    def create_scenario_summary(self, scenario_results: List[ScenarioResult]) -> pd.DataFrame:
        """
        Create a summary DataFrame of scenario results.
        
        Args:
            scenario_results: List of scenario results
            
        Returns:
            DataFrame with scenario summary
        """
        if not scenario_results:
            return pd.DataFrame()
        
        data = []
        for result in scenario_results:
            pnl = result.pnl_components
            data.append({
                'scenario_id': result.scenario_id,
                'spot_shock': pnl.spot_shock,
                'vol_shock': pnl.vol_shock,
                'time_decay_days': pnl.time_decay,
                'delta_pnl': pnl.delta_pnl,
                'gamma_pnl': pnl.gamma_pnl,
                'theta_pnl': pnl.theta_pnl,
                'vega_pnl': pnl.vega_pnl,
                'total_pnl': pnl.total_pnl,
                'pnl_percentage': result.pnl_percentage,
                'original_price': pnl.original_price,
                'new_theoretical_price': pnl.new_theoretical_price
            })
        
        df = pd.DataFrame(data)
        
        # Sort by total PnL for easier analysis
        df = df.sort_values('total_pnl', ascending=False).reset_index(drop=True)
        
        return df

    def stress_test(self,
                   option_params: OptionParameters,
                   extreme_scenarios: bool = True) -> Dict[str, Any]:
        """
        Perform stress testing with extreme market scenarios.
        
        Args:
            option_params: Option parameters
            extreme_scenarios: Whether to include extreme stress scenarios
            
        Returns:
            Dictionary with stress test results
        """
        logger.info("Performing stress test analysis...")
        
        # Define stress scenarios
        if extreme_scenarios:
            stress_params = ScenarioParameters(
                spot_shocks=[-0.5, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.5],  # Up to 50% moves
                vol_shocks=[-0.8, -0.5, -0.2, 0, 0.2, 0.5, 0.8, 1.0],  # Up to 100% vol changes
                time_decay_days=[0, 1, 7, 14, 30, 60, 90]  # Extended time horizons
            )
        else:
            stress_params = ScenarioParameters()  # Default scenarios
        
        # Run scenarios
        results = self.analyze_scenarios(option_params, stress_params)
        
        # Calculate metrics
        risk_metrics = self.calculate_risk_metrics(results)
        summary_df = self.create_scenario_summary(results)
        
        # Find worst-case scenarios
        worst_case = min(results, key=lambda r: r.pnl_components.total_pnl)
        best_case = max(results, key=lambda r: r.pnl_components.total_pnl)
        
        return {
            'scenario_results': results,
            'risk_metrics': risk_metrics,
            'summary_dataframe': summary_df,
            'worst_case_scenario': worst_case,
            'best_case_scenario': best_case,
            'total_scenarios': len(results),
            'extreme_scenarios_included': extreme_scenarios
        }

    def portfolio_analysis(self, options_list: List[OptionParameters],
                          scenario_params: Optional[ScenarioParameters] = None) -> Dict[str, Any]:
        """
        Analyze PnL for a portfolio of options.
        
        Args:
            options_list: List of option parameters representing the portfolio
            scenario_params: Scenario parameters for analysis
            
        Returns:
            Portfolio analysis results
        """
        logger.info(f"Analyzing portfolio of {len(options_list)} options...")
        
        if scenario_params is None:
            scenario_params = ScenarioParameters()
        
        # Analyze each option individually
        individual_results = {}
        for i, option_params in enumerate(options_list):
            option_id = f"Option_{i+1}"
            individual_results[option_id] = self.analyze_scenarios(option_params, scenario_params)
        
        # Calculate portfolio-level metrics for each scenario combination
        portfolio_scenarios = []
        
        # Get first option's scenarios to establish scenario structure
        first_option_scenarios = list(individual_results.values())[0]
        
        for scenario_idx, base_scenario in enumerate(first_option_scenarios):
            portfolio_pnl = 0.0
            portfolio_components = {
                'delta_pnl': 0.0,
                'gamma_pnl': 0.0, 
                'theta_pnl': 0.0,
                'vega_pnl': 0.0
            }
            
            # Sum PnL across all options for this scenario
            for option_results in individual_results.values():
                if scenario_idx < len(option_results):
                    scenario_result = option_results[scenario_idx]
                    pnl = scenario_result.pnl_components
                    
                    portfolio_pnl += pnl.total_pnl
                    portfolio_components['delta_pnl'] += pnl.delta_pnl
                    portfolio_components['gamma_pnl'] += pnl.gamma_pnl
                    portfolio_components['theta_pnl'] += pnl.theta_pnl
                    portfolio_components['vega_pnl'] += pnl.vega_pnl
            
            portfolio_scenarios.append({
                'scenario_id': base_scenario.scenario_id,
                'total_pnl': portfolio_pnl,
                **portfolio_components,
                'spot_shock': base_scenario.pnl_components.spot_shock,
                'vol_shock': base_scenario.pnl_components.vol_shock,
                'time_decay': base_scenario.pnl_components.time_decay
            })
        
        # Calculate portfolio risk metrics
        portfolio_pnls = [s['total_pnl'] for s in portfolio_scenarios]
        portfolio_risk_metrics = {
            'mean_pnl': np.mean(portfolio_pnls),
            'std_pnl': np.std(portfolio_pnls),
            'min_pnl': np.min(portfolio_pnls),
            'max_pnl': np.max(portfolio_pnls),
            'var_95': np.percentile(portfolio_pnls, 5),
            'var_99': np.percentile(portfolio_pnls, 1),
            'prob_loss': len([p for p in portfolio_pnls if p < 0]) / len(portfolio_pnls) * 100
        }
        
        return {
            'individual_results': individual_results,
            'portfolio_scenarios': portfolio_scenarios,
            'portfolio_risk_metrics': portfolio_risk_metrics,
            'num_options': len(options_list),
            'total_scenarios': len(portfolio_scenarios)
        }

# Convenience functions for quick analysis
def quick_pnl_analysis(spot_price: float,
                      strike_price: float, 
                      time_to_expiry: float,
                      volatility: float,
                      option_type: str = "call",
                      spot_shocks: Optional[List[float]] = None,
                      vol_shocks: Optional[List[float]] = None) -> pd.DataFrame:
    """
    Quick PnL analysis for a single option.
    
    Args:
        spot_price: Current underlying price
        strike_price: Strike price
        time_to_expiry: Time to expiry in years
        volatility: Implied volatility (decimal)
        option_type: 'call' or 'put'
        spot_shocks: List of spot price shocks (default: [-10%, 0%, +10%])
        vol_shocks: List of volatility shocks (default: [-20%, 0%, +20%])
        
    Returns:
        DataFrame with PnL analysis results
    """
    # Create option parameters
    params = OptionParameters(
        spot_price=spot_price,
        strike_price=strike_price,
        time_to_expiry=time_to_expiry,
        volatility=volatility,
        risk_free_rate=0.05,
        option_type=OptionType.CALL if option_type.lower() == 'call' else OptionType.PUT
    )
    
    # Create scenario parameters
    scenario_params = ScenarioParameters(
        spot_shocks=spot_shocks or [-0.1, 0.0, 0.1],
        vol_shocks=vol_shocks or [-0.2, 0.0, 0.2],
        time_decay_days=[0, 1, 7]
    )
    
    # Run analysis
    pnl_simulator = TaylorExpansionPnL()
    results = pnl_simulator.analyze_scenarios(params, scenario_params)
    
    return pnl_simulator.create_scenario_summary(results)

# Testing and validation
def test_taylor_expansion_pnl():
    """
    Test Taylor expansion PnL calculations.
    
    Returns:
        True if all tests pass
    """
    try:
        logger.info("Testing Taylor expansion PnL simulator...")
        
        # Create test option
        test_params = OptionParameters(
            spot_price=100.0,
            strike_price=100.0,
            time_to_expiry=0.25,  # 3 months
            volatility=0.20,      # 20% vol
            risk_free_rate=0.05,
            option_type=OptionType.CALL
        )
        
        # Initialize simulator
        pnl_sim = TaylorExpansionPnL()
        
        # Test 1: Basic PnL calculation
        pnl_components = pnl_sim.calculate_pnl_components(
            test_params,
            spot_shock=0.1,    # +10% spot move
            vol_shock=0.1,     # +10% vol move
            time_decay_days=1  # 1 day decay
        )
        
        # Validate results
        assert pnl_components.delta_pnl > 0, "Call delta PnL should be positive for upward spot move"
        assert pnl_components.gamma_pnl > 0, "Gamma PnL should be positive (convexity)"
        assert pnl_components.theta_pnl < 0, "Theta PnL should be negative (time decay)"
        assert pnl_components.vega_pnl > 0, "Vega PnL should be positive for vol increase"
        
        # Test 2: Scenario analysis
        results = pnl_sim.analyze_scenarios(test_params)
        assert len(results) > 0, "Should generate scenario results"
        
        # Test 3: Risk metrics
        risk_metrics = pnl_sim.calculate_risk_metrics(results)
        assert 'var_95_pnl' in risk_metrics, "Should calculate VaR"
        assert 'prob_loss' in risk_metrics, "Should calculate probability of loss"
        
        # Test 4: Summary DataFrame
        summary_df = pnl_sim.create_scenario_summary(results)
        assert not summary_df.empty, "Should create non-empty summary"
        assert 'total_pnl' in summary_df.columns, "Should have total_pnl column"
        
        logger.info("✅ All Taylor expansion PnL tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Taylor expansion PnL tests failed: {e}")
        return False

if __name__ == "__main__":
    # Run tests when module is executed directly
    test_taylor_expansion_pnl()
