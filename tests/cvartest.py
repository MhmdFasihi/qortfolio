# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com
# Test CVaR implementation
from src.analytics.pnl_simulator import TaylorExpansionPnL
from src.models.black_scholes import OptionParameters, OptionType

# Create test option
params = OptionParameters(
    spot_price=30000, strike_price=32000, time_to_expiry=30/365.25,
    volatility=0.80, risk_free_rate=0.05, option_type=OptionType.CALL
)

# Run analysis
pnl_sim = TaylorExpansionPnL()
results = pnl_sim.analyze_scenarios(params)
risk_metrics = pnl_sim.calculate_risk_metrics(results)

# Check what's available
print("Available risk metrics:")
for key, value in risk_metrics.items():
    print(f"  {key}: {value}")