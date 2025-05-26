"""
Analytics Package for Bitcoin Options Analytics Platform

This package provides advanced analytics capabilities including:
- Taylor Expansion PnL Analysis (Primary Feature)
- Scenario Analysis and Stress Testing  
- Portfolio Risk Management
- Performance Attribution
"""

__all__ = [
    "TaylorExpansionPnL",
    "ScenarioParameters", 
    "PnLComponents",
    "ScenarioResult",
    "quick_pnl_analysis"
]

# Import main classes when available
try:
    from .pnl_simulator import (
        TaylorExpansionPnL,
        ScenarioParameters,
        PnLComponents, 
        ScenarioResult,
        quick_pnl_analysis
    )
except ImportError:
    # Graceful fallback during development
    TaylorExpansionPnL = None
    ScenarioParameters = None
    PnLComponents = None
    ScenarioResult = None
    quick_pnl_analysis = None

# Package metadata
__version__ = "1.0.0"
__author__ = "Bitcoin Options Analytics Platform"