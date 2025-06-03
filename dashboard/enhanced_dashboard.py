"""
Bitcoin Options Analytics Platform - ENHANCED COMPLETE DASHBOARD

This dashboard integrates the comprehensive visualization framework with real-time
data collection and Taylor expansion PnL analysis.

NEW FEATURES INTEGRATED:
- Enhanced Visualization Framework
- Taylor Expansion PnL Heatmaps  
- Greeks Risk Dashboard
- Advanced Market Analysis
- Strategy Analysis Tools
- Portfolio Management
- Real-time Data Integration

Usage:
    streamlit run dashboard/app.py
"""

import sys
import logging
import traceback
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any, List

# Add src to Python path
current_dir = Path(__file__).parent
root_dir = current_dir.parent
src_dir = root_dir / "src"
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(src_dir))

# Import statements with error handling
try:
    import streamlit as st
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    import plotly.express as px
    import requests
    import time
    STREAMLIT_AVAILABLE = True
except ImportError as e:
    print(f"❌ Import error: {e}")
    STREAMLIT_AVAILABLE = False

# Test backend availability
def test_backend_availability():
    """Test if all backend modules are available."""
    try:
        # Core modules
        from src.analytics.pnl_simulator import TaylorExpansionPnL, ScenarioParameters
        from src.models.black_scholes import BlackScholesModel, OptionParameters, OptionType
        from src.data.collectors import DeribitCollector
        from src import __version__
        
        # NEW: Enhanced visualization framework
        from src.visualization.enhanced_viz_framework import (
            EnhancedOptionsVisualizer, 
            VisualizationType, 
            VisualizationConfig,
            create_visualizer,
            quick_pnl_heatmap
        )
        
        return True, None
    except Exception as e:
        return False, str(e)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# MARKET DATA HELPERS (Enhanced)
# ==========================================

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_current_crypto_price(currency: str = "BTC") -> float:
    """Get current cryptocurrency price from CoinGecko API."""
    try:
        currency_map = {'BTC': 'bitcoin', 'ETH': 'ethereum'}
        coin_id = currency_map.get(currency.upper(), 'bitcoin')
        url = f"https://api.coingecko.com/api/v3/simple/price"
        
        response = requests.get(url, params={'ids': coin_id, 'vs_currencies': 'usd'}, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        price = data[coin_id]['usd']
        logger.info(f"✅ Current {currency} price: ${price:,.2f}")
        return float(price)
        
    except Exception as e:
        logger.warning(f"⚠️ Could not fetch {currency} price: {e}")
        fallback_prices = {'BTC': 83000.0, 'ETH': 2600.0}
        return fallback_prices.get(currency.upper(), 30000.0)

@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_available_options_data(currency: str = "BTC") -> pd.DataFrame:
    """Get sample options data for smart defaults and visualization."""
    try:
        from src.data.collectors import DeribitCollector
        
        end_date = date.today() - timedelta(days=1)
        start_date = end_date - timedelta(days=2)
        
        logger.info(f"🔍 Fetching {currency} options for analysis...")
        
        with DeribitCollector() as collector:
            data = collector.collect_options_data(
                currency=currency,
                start_date=start_date,
                end_date=end_date,
                max_collection_time=15,
                max_total_records=1000
            )
            
            if not data.empty:
                logger.info(f"✅ Found {len(data)} {currency} options")
                return data
            else:
                logger.warning(f"⚠️ No {currency} options data found")
                return pd.DataFrame()
                
    except Exception as e:
        logger.warning(f"⚠️ Could not fetch {currency} options data: {e}")
        return pd.DataFrame()

def get_smart_defaults(currency: str = "BTC") -> dict:
    """Get intelligent defaults based on real market data."""
    try:
        current_price = get_current_crypto_price(currency)
        options_data = get_available_options_data(currency)
        
        # Calculate intelligent strike price
        if not options_data.empty and 'strike_price' in options_data.columns:
            strikes = options_data['strike_price'].unique()
            nearest_strike = min(strikes, key=lambda x: abs(x - current_price))
        else:
            increment = 5000 if current_price >= 50000 else 2000 if current_price >= 20000 else 1000
            nearest_strike = ((current_price // increment) + 1) * increment
        
        # Calculate intelligent maturity
        if not options_data.empty and 'time_to_maturity' in options_data.columns:
            ttm_days = options_data['time_to_maturity'] * 365.25
            valid_days = ttm_days[ttm_days > 1]
            nearest_maturity_days = valid_days.min() if not valid_days.empty else 30.0
        else:
            nearest_maturity_days = 30.0
        
        # Calculate intelligent volatility
        if not options_data.empty and 'implied_volatility' in options_data.columns:
            median_iv = options_data['implied_volatility'].median() * 100
            default_volatility = max(20.0, min(200.0, median_iv))
        else:
            default_volatility = {'BTC': 80.0, 'ETH': 90.0}.get(currency, 80.0)
        
        return {
            'spot_price': current_price,
            'strike_price': nearest_strike,
            'time_to_expiry_days': nearest_maturity_days,
            'volatility_percent': default_volatility,
            'risk_free_rate_percent': 5.0
        }
        
    except Exception as e:
        logger.error(f"❌ Error calculating smart defaults: {e}")
        return {
            'spot_price': 30000.0, 'strike_price': 32000.0, 'time_to_expiry_days': 30.0,
            'volatility_percent': 80.0, 'risk_free_rate_percent': 5.0
        }

def display_market_data_status(currency: str = "BTC"):
    """Display current market data status in sidebar."""
    try:
        with st.sidebar:
            st.markdown("---")
            st.subheader("📊 Live Market Data")
            
            current_price = get_current_crypto_price(currency)
            st.metric(f"{currency} Price", f"${current_price:,.2f}")
            st.caption(f"Updated: {datetime.now().strftime('%H:%M:%S')}")
            
            if st.button("🔄 Refresh Market Data"):
                st.cache_data.clear()
                st.rerun()
                
    except Exception as e:
        logger.warning(f"Could not display market data status: {e}")

# ==========================================
# MAIN DASHBOARD CREATION
# ==========================================

def create_enhanced_dashboard():
    """Create the main enhanced dashboard with full visualization integration."""
    
    # Test backend availability
    backend_available, error_msg = test_backend_availability()
    
    if not backend_available:
        st.error("❌ **Backend modules not available**")
        st.error(f"Error: {error_msg}")
        st.info("💡 **Solutions:**")
        st.info("1. Make sure you're in the project root directory")
        st.info("2. Install dependencies: `pip install -r requirements.txt`")
        st.info("3. Check that src/ directory contains all modules")
        return
    
    # Import all backend modules
    from src.analytics.pnl_simulator import TaylorExpansionPnL, ScenarioParameters
    from src.models.black_scholes import BlackScholesModel, OptionParameters, OptionType
    from src.data.collectors import DeribitCollector
    from src import __version__
    from src.visualization.enhanced_viz_framework import (
        EnhancedOptionsVisualizer, VisualizationType, VisualizationConfig,
        create_visualizer, quick_pnl_heatmap
    )
    
    # Page configuration
    st.set_page_config(
        page_title="Bitcoin Options Analytics Platform",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("🚀 Bitcoin Options Analytics Platform")
    st.markdown(f"**Version:** {__version__} | **Status:** ✅ Enhanced with Comprehensive Visualizations")
    
    # Enhanced sidebar navigation
    st.sidebar.title("📋 Navigation")
    
    # Main categories
    analysis_category = st.sidebar.selectbox(
        "Choose Analysis Category:",
        [
            "🎯 Core Analytics",
            "📊 Advanced Visualizations", 
            "📈 Market Analysis",
            "⚡ Greeks & Risk",
            "🔧 Data & System"
        ]
    )
    
    # Sub-navigation based on category
    if analysis_category == "🎯 Core Analytics":
        page = st.sidebar.selectbox(
            "Core Analysis Tools:",
            [
                "🎯 Taylor Expansion PnL",
                "📈 Scenario Analysis", 
                "🧮 Greeks Calculator",
                "📊 Strategy Analysis"
            ]
        )
    elif analysis_category == "📊 Advanced Visualizations":
        page = st.sidebar.selectbox(
            "Visualization Tools:",
            [
                "🌊 Taylor PnL Heatmaps",
                "📊 Greeks Risk Dashboard",
                "📈 PnL Components Analysis",
                "🎛️ Interactive Parameter Explorer"
            ]
        )
    elif analysis_category == "📈 Market Analysis":
        page = st.sidebar.selectbox(
            "Market Analysis Tools:",
            [
                "🌋 Volatility Surface",
                "📊 IV Skew Analysis", 
                "📈 IV Timeseries",
                "📊 Option Distributions",
                "📊 Market Intelligence"
            ]
        )
    elif analysis_category == "⚡ Greeks & Risk":
        page = st.sidebar.selectbox(
            "Risk Management:",
            [
                "⚡ Greeks 3D Analysis",
                "📊 Portfolio Greeks",
                "🚨 Risk Monitoring",
                "🎯 Sensitivity Analysis"
            ]
        )
    else:  # Data & System
        page = st.sidebar.selectbox(
            "Data & System:",
            [
                "📊 Data Collection",
                "ℹ️ System Status",
                "🧪 Testing Tools"
            ]
        )
    
    # Route to appropriate page
    if page == "🎯 Taylor Expansion PnL":
        create_taylor_pnl_page()
    elif page == "🌊 Taylor PnL Heatmaps":
        create_pnl_heatmap_page()
    elif page == "📊 Greeks Risk Dashboard":
        create_greeks_dashboard_page()
    elif page == "📈 PnL Components Analysis":
        create_pnl_components_page()
    elif page == "🎛️ Interactive Parameter Explorer":
        create_parameter_explorer_page()
    elif page == "🌋 Volatility Surface":
        create_volatility_surface_page()
    elif page == "📊 IV Skew Analysis":
        create_iv_skew_page()
    elif page == "📈 IV Timeseries":
        create_iv_timeseries_page()
    elif page == "📊 Option Distributions":
        create_distributions_page()
    elif page == "📊 Market Intelligence":
        create_market_intelligence_page()
    elif page == "⚡ Greeks 3D Analysis":
        create_greeks_3d_page()
    elif page == "📊 Portfolio Greeks":
        create_portfolio_greeks_page()
    elif page == "🚨 Risk Monitoring":
        create_risk_monitoring_page()
    elif page == "🎯 Sensitivity Analysis":
        create_sensitivity_analysis_page()
    elif page == "📈 Scenario Analysis":
        create_scenario_analysis_page()
    elif page == "🧮 Greeks Calculator":
        create_greeks_calculator_page()
    elif page == "📊 Strategy Analysis":
        create_strategy_analysis_page()
    elif page == "📊 Data Collection":
        create_data_collection_page()
    elif page == "ℹ️ System Status":
        create_system_status_page()
    elif page == "🧪 Testing Tools":
        create_testing_tools_page()

# ==========================================
# ENHANCED VISUALIZATION PAGES
# ==========================================

def create_pnl_heatmap_page():
    """Create Taylor expansion PnL heatmap visualization page."""
    from src.analytics.pnl_simulator import TaylorExpansionPnL
    from src.models.black_scholes import OptionParameters, OptionType
    from src.visualization.enhanced_viz_framework import create_visualizer
    
    st.header("🌊 Taylor Expansion PnL Heatmaps")
    st.markdown("**Interactive 2D/3D PnL Analysis:** Visualize PnL across spot price and volatility parameter space")
    
    # Currency selection
    col1, col2 = st.columns([3, 1])
    with col1:
        currency = st.selectbox("Currency", ["BTC", "ETH"], key="heatmap_currency")
    with col2:
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
    
    # Get smart defaults
    smart_defaults = get_smart_defaults(currency)
    display_market_data_status(currency)
    
    # Parameter inputs
    st.subheader("📊 Option Parameters")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        spot = st.number_input("Spot Price ($)", value=smart_defaults['spot_price'], min_value=1000.0)
        strike = st.number_input("Strike Price ($)", value=smart_defaults['strike_price'], min_value=1000.0)
    
    with col2:
        tte_days = st.number_input("Days to Expiry", value=smart_defaults['time_to_expiry_days'], min_value=1.0)
        vol = st.slider("Volatility (%)", 10.0, 200.0, smart_defaults['volatility_percent']) / 100.0
    
    with col3:
        opt_type = st.selectbox("Option Type", ["Call", "Put"])
        time_decay = st.number_input("Time Decay (days)", value=1.0, min_value=0.0, max_value=30.0)
    
    with col4:
        spot_range_pct = st.slider("Spot Range (±%)", 10, 50, 20)
        vol_range_pct = st.slider("Vol Range (±%)", 10, 100, 30)
    
    if st.button("🌊 Generate PnL Heatmap", type="primary"):
        try:
            with st.spinner("Generating Taylor expansion PnL heatmap..."):
                # Create option parameters
                params = OptionParameters(
                    spot_price=spot,
                    strike_price=strike,
                    time_to_expiry=tte_days / 365.25,
                    volatility=vol,
                    risk_free_rate=0.05,
                    option_type=OptionType.CALL if opt_type == "Call" else OptionType.PUT
                )
                
                # Create visualizer
                visualizer = create_visualizer(currency=currency)
                
                # Generate heatmap
                spot_range = (1 - spot_range_pct/100, 1 + spot_range_pct/100)
                vol_range = (-vol_range_pct/100, vol_range_pct/100)
                
                fig = visualizer.plot_taylor_pnl_heatmap(
                    params,
                    spot_range=spot_range,
                    vol_range=vol_range,
                    time_decay_days=time_decay
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Analysis summary
                st.success("✅ **Taylor Expansion PnL Heatmap Generated!**")
                st.info("💡 **Interpretation Guide:**")
                st.markdown("""
                - **Red areas**: Potential losses
                - **Green areas**: Potential gains  
                - **Gradients**: PnL sensitivity to parameter changes
                - **Component breakdown**: See individual Greek contributions
                """)
                
        except Exception as e:
            st.error(f"❌ **Heatmap generation failed:** {str(e)}")
            with st.expander("🔍 Error Details"):
                st.code(traceback.format_exc())

def create_greeks_dashboard_page():
    """Create Greeks risk monitoring dashboard."""
    from src.models.black_scholes import OptionParameters, OptionType
    from src.visualization.enhanced_viz_framework import create_visualizer
    
    st.header("📊 Greeks Risk Dashboard")
    st.markdown("**Real-time Greeks monitoring with risk alerts and gauges**")
    
    # Currency and parameters
    currency = st.selectbox("Currency", ["BTC", "ETH"], key="greeks_dash_currency")
    smart_defaults = get_smart_defaults(currency)
    display_market_data_status(currency)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Option Setup")
        spot = st.number_input("Spot Price", value=smart_defaults['spot_price'], key="dash_spot")
        strike = st.number_input("Strike Price", value=smart_defaults['strike_price'], key="dash_strike")
        tte = st.number_input("Days to Expiry", value=smart_defaults['time_to_expiry_days'], key="dash_tte")
    
    with col2:
        st.subheader("⚠️ Risk Thresholds")
        delta_threshold = st.slider("Delta Alert Level", 0.1, 1.0, 0.5)
        gamma_threshold = st.slider("Gamma Alert Level", 0.01, 0.2, 0.1)
        theta_threshold = st.number_input("Theta Alert ($)", value=-50.0)
        vega_threshold = st.number_input("Vega Alert ($)", value=100.0)
    
    vol = st.slider("Current Volatility (%)", 10.0, 200.0, smart_defaults['volatility_percent'], key="dash_vol") / 100.0
    opt_type = st.selectbox("Option Type", ["Call", "Put"], key="dash_type")
    
    if st.button("📊 Launch Greeks Dashboard", type="primary"):
        try:
            with st.spinner("Creating Greeks risk dashboard..."):
                # Create parameters
                params = OptionParameters(
                    spot_price=spot,
                    strike_price=strike,
                    time_to_expiry=tte / 365.25,
                    volatility=vol,
                    risk_free_rate=0.05,
                    option_type=OptionType.CALL if opt_type == "Call" else OptionType.PUT
                )
                
                # Risk thresholds
                thresholds = {
                    'delta': delta_threshold,
                    'gamma': gamma_threshold,
                    'theta': theta_threshold,
                    'vega': vega_threshold
                }
                
                # Create dashboard
                visualizer = create_visualizer(currency=currency)
                fig = visualizer.plot_greeks_risk_dashboard(params, thresholds)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Risk alerts
                from src.models.black_scholes import BlackScholesModel
                bs_model = BlackScholesModel()
                greeks = bs_model.calculate_greeks(params)
                
                st.subheader("🚨 Risk Alert Summary")
                
                alerts = []
                if abs(greeks.delta) > delta_threshold:
                    alerts.append(f"⚠️ **Delta Alert**: {greeks.delta:.4f} exceeds threshold {delta_threshold}")
                if greeks.gamma > gamma_threshold:
                    alerts.append(f"⚠️ **Gamma Alert**: {greeks.gamma:.6f} exceeds threshold {gamma_threshold}")
                if greeks.theta < theta_threshold:
                    alerts.append(f"⚠️ **Theta Alert**: ${greeks.theta:.2f} below threshold ${theta_threshold}")
                if abs(greeks.vega) > vega_threshold:
                    alerts.append(f"⚠️ **Vega Alert**: ${greeks.vega:.2f} exceeds threshold ${vega_threshold}")
                
                if alerts:
                    for alert in alerts:
                        st.warning(alert)
                else:
                    st.success("✅ **All Greeks within risk thresholds**")
                
        except Exception as e:
            st.error(f"❌ **Dashboard creation failed:** {str(e)}")

def create_pnl_components_page():
    """Create detailed PnL components breakdown analysis."""
    from src.analytics.pnl_simulator import TaylorExpansionPnL, ScenarioParameters
    from src.models.black_scholes import OptionParameters, OptionType
    from src.visualization.enhanced_viz_framework import create_visualizer
    
    st.header("📈 PnL Components Analysis")
    st.markdown("**Detailed breakdown of Taylor expansion components:** `ΔC ≈ δΔS + ½γ(ΔS)² + θΔt + νΔσ`")
    
    # Setup
    currency = st.selectbox("Currency", ["BTC", "ETH"], key="components_currency")
    smart_defaults = get_smart_defaults(currency)
    display_market_data_status(currency)
    
    # Parameters
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Option Parameters")
        spot = st.number_input("Spot Price", value=smart_defaults['spot_price'], key="comp_spot")
        strike = st.number_input("Strike Price", value=smart_defaults['strike_price'], key="comp_strike")
        tte = st.number_input("Days to Expiry", value=smart_defaults['time_to_expiry_days'], key="comp_tte")
        vol = st.slider("Volatility (%)", 10.0, 200.0, smart_defaults['volatility_percent'], key="comp_vol") / 100.0
        opt_type = st.selectbox("Option Type", ["Call", "Put"], key="comp_type")
    
    with col2:
        st.subheader("⚡ Scenario Parameters")
        spot_shocks = st.multiselect(
            "Spot Price Shocks (%)",
            [-20, -10, -5, 0, 5, 10, 20],
            default=[-10, -5, 0, 5, 10]
        )
        vol_shocks = st.multiselect(
            "Volatility Shocks (%)", 
            [-30, -20, -10, 0, 10, 20, 30],
            default=[-20, -10, 0, 10, 20]
        )
        time_decay = st.number_input("Time Decay (days)", value=1.0, key="comp_time")
    
    if st.button("📈 Analyze PnL Components", type="primary"):
        try:
            with st.spinner("Analyzing PnL component breakdown..."):
                # Create parameters
                params = OptionParameters(
                    spot_price=spot,
                    strike_price=strike,
                    time_to_expiry=tte / 365.25,
                    volatility=vol,
                    risk_free_rate=0.05,
                    option_type=OptionType.CALL if opt_type == "Call" else OptionType.PUT
                )
                
                # Convert shocks to decimals
                spot_shocks_decimal = [s/100 for s in spot_shocks]
                vol_shocks_decimal = [v/100 for v in vol_shocks]
                
                # Create visualizer and generate analysis
                visualizer = create_visualizer(currency=currency)
                fig = visualizer.plot_pnl_components_breakdown(
                    params,
                    spot_shocks=spot_shocks_decimal,
                    vol_shocks=vol_shocks_decimal,
                    time_decay_days=time_decay
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Component explanations
                st.subheader("📚 Component Explanations")
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown("""
                    **🔵 Delta (δΔS):**
                    - Linear price sensitivity
                    - Dominates for small moves
                    - Directional exposure
                    
                    **🟢 Gamma (½γ(ΔS)²):**
                    - Curvature/convexity benefit
                    - Always positive (long options)
                    - Accelerates with larger moves
                    """)
                
                with col_b:
                    st.markdown("""
                    **🔴 Theta (θΔt):**
                    - Time decay cost
                    - Generally negative (long options)
                    - Accelerates near expiration
                    
                    **🟠 Vega (νΔσ):**
                    - Volatility sensitivity
                    - Positive for long options
                    - Higher for ATM options
                    """)
                
        except Exception as e:
            st.error(f"❌ **Component analysis failed:** {str(e)}")

def create_parameter_explorer_page():
    """Create interactive parameter exploration tool."""
    from src.analytics.pnl_simulator import TaylorExpansionPnL
    from src.models.black_scholes import BlackScholesModel, OptionParameters, OptionType
    
    st.header("🎛️ Interactive Parameter Explorer")
    st.markdown("**Real-time parameter sensitivity analysis with live updates**")
    
    currency = st.selectbox("Currency", ["BTC", "ETH"], key="explorer_currency")
    smart_defaults = get_smart_defaults(currency)
    display_market_data_status(currency)
    
    # Real-time parameter controls
    st.subheader("🎛️ Live Parameter Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        spot = st.slider("Spot Price", 
                        float(smart_defaults['spot_price'] * 0.5), 
                        float(smart_defaults['spot_price'] * 1.5), 
                        smart_defaults['spot_price'], 
                        step=500.0)
        
        strike = st.slider("Strike Price",
                          float(smart_defaults['strike_price'] * 0.5),
                          float(smart_defaults['strike_price'] * 1.5),
                          smart_defaults['strike_price'],
                          step=500.0)
    
    with col2:
        tte_days = st.slider("Days to Expiry", 1.0, 365.0, smart_defaults['time_to_expiry_days'])
        vol = st.slider("Volatility (%)", 10.0, 200.0, smart_defaults['volatility_percent']) / 100.0
    
    with col3:
        opt_type = st.selectbox("Option Type", ["Call", "Put"])
        spot_shock = st.slider("Spot Shock (%)", -50.0, 50.0, 0.0) / 100.0
        vol_shock = st.slider("Vol Shock (%)", -50.0, 100.0, 0.0) / 100.0
        time_decay = st.slider("Time Decay (days)", 0.0, 30.0, 1.0)
    
    # Real-time calculations
    try:
        # Create models
        bs_model = BlackScholesModel()
        pnl_sim = TaylorExpansionPnL()
        
        # Create parameters
        params = OptionParameters(
            spot_price=spot,
            strike_price=strike,
            time_to_expiry=tte_days / 365.25,
            volatility=vol,
            risk_free_rate=0.05,
            option_type=OptionType.CALL if opt_type == "Call" else OptionType.PUT
        )
        
        # Calculate Greeks and PnL
        greeks = bs_model.calculate_greeks(params)
        pnl_components = pnl_sim.calculate_pnl_components(
            params, spot_shock, vol_shock, time_decay
        )
        
        # Display results in real-time
        st.subheader("📊 Live Results")
        
        # Metrics row 1
        met_col1, met_col2, met_col3, met_col4 = st.columns(4)
        
        with met_col1:
            st.metric("Option Price", f"${greeks.option_price:.2f}")
            st.metric("Delta", f"{greeks.delta:.4f}")
        
        with met_col2:
            st.metric("Gamma", f"{greeks.gamma:.6f}")
            st.metric("Theta (daily)", f"${greeks.theta:.4f}")
        
        with met_col3:
            st.metric("Vega", f"${greeks.vega:.4f}")
            st.metric("Moneyness", f"{spot/strike:.4f}")
        
        with met_col4:
            st.metric("Total PnL", f"${pnl_components.total_pnl:.2f}",
                     f"{(pnl_components.total_pnl/greeks.option_price*100) if greeks.option_price > 0 else 0:.1f}%")
            st.metric("Time Value", f"{tte_days:.0f} days")
        
        # PnL breakdown
        st.subheader("⚡ PnL Component Breakdown")
        
        pnl_col1, pnl_col2, pnl_col3, pnl_col4 = st.columns(4)
        
        with pnl_col1:
            st.metric("Delta PnL", f"${pnl_components.delta_pnl:.2f}")
        with pnl_col2:
            st.metric("Gamma PnL", f"${pnl_components.gamma_pnl:.2f}")
        with pnl_col3:
            st.metric("Theta PnL", f"${pnl_components.theta_pnl:.2f}")
        with pnl_col4:
            st.metric("Vega PnL", f"${pnl_components.vega_pnl:.2f}")
        
        # Live chart
        components = ["Delta", "Gamma", "Theta", "Vega"]
        values = [pnl_components.delta_pnl, pnl_components.gamma_pnl, 
                 pnl_components.theta_pnl, pnl_components.vega_pnl]
        colors = ["blue", "green", "red", "orange"]
        
        fig = go.Figure(data=[
            go.Bar(x=components, y=values, marker_color=colors,
                  text=[f"${v:.2f}" for v in values], textposition='auto')
        ])
        
        fig.update_layout(
            title="Live PnL Component Breakdown",
            xaxis_title="Component",
            yaxis_title="PnL ($)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ **Live calculation error:** {str(e)}")

# ==========================================
# ENHANCED EXISTING PAGES
# ==========================================

def create_taylor_pnl_page():
    """Enhanced Taylor expansion PnL analysis page."""
    from src.analytics.pnl_simulator import TaylorExpansionPnL
    from src.models.black_scholes import OptionParameters, OptionType
    
    st.header("🎯 Taylor Expansion PnL Analysis")
    st.markdown("**Primary Feature:** `ΔC ≈ δΔS + ½γ(ΔS)² + θΔt + νΔσ`")
    
    # Enhanced currency selection with refresh
    currency_col, refresh_col, viz_col = st.columns([3, 1, 1])
    
    with currency_col:
        selected_currency = st.selectbox("Select Currency", ["BTC", "ETH"], index=0)
    
    with refresh_col:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
    
    with viz_col:
        st.markdown("<br>", unsafe_allow_html=True)
        show_advanced = st.checkbox("🎨 Advanced Viz")
    
    # Get smart defaults
    with st.spinner(f"Loading real market data for {selected_currency}..."):
        smart_defaults = get_smart_defaults(selected_currency)
    
    display_market_data_status(selected_currency)
    
    # Create columns for input parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📊 Option Parameters")
        spot_price = st.number_input(f"Current {selected_currency} Price ($)", 
                                   min_value=1000.0, max_value=200000.0,
                                   value=smart_defaults['spot_price'], step=1000.0)
        
        strike_price = st.number_input("Strike Price ($)",
                                     min_value=1000.0, max_value=200000.0,
                                     value=smart_defaults['strike_price'], step=1000.0)
        
        time_to_expiry = st.number_input("Time to Expiry (days)",
                                       min_value=1.0, max_value=365.0,
                                       value=smart_defaults['time_to_expiry_days'], step=1.0)
    
    with col2:
        st.subheader("📈 Market Parameters")
        volatility = st.slider("Implied Volatility (%)", min_value=10.0, max_value=200.0,
                              value=smart_defaults['volatility_percent'], step=5.0) / 100.0
        
        risk_free_rate = st.slider("Risk-Free Rate (%)", min_value=0.0, max_value=10.0,
                                 value=smart_defaults['risk_free_rate_percent'], step=0.1) / 100.0
        
        option_type = st.selectbox("Option Type", ["Call", "Put"])
    
    with col3:
        st.subheader("⚡ Scenario Shocks")
        spot_shock = st.slider("Spot Price Shock (%)", min_value=-50.0, max_value=50.0,
                              value=10.0, step=1.0) / 100.0
        
        vol_shock = st.slider("Volatility Shock (%)", min_value=-50.0, max_value=100.0,
                             value=20.0, step=5.0) / 100.0
        
        time_decay = st.number_input("Time Decay (days)", min_value=0.0, max_value=30.0,
                                   value=1.0, step=0.5)
    
    # Show data source info
    st.info(f"💡 **Smart Defaults Active:** Using real market data for {selected_currency}. " +
           f"Current price: ${smart_defaults['spot_price']:,.0f}, " +
           f"Nearest strike: ${smart_defaults['strike_price']:,.0f}")
    
    # Calculate PnL button
    if st.button("🚀 **Calculate Taylor Expansion PnL**", type="primary"):
        try:
            with st.spinner("Calculating PnL components..."):
                # Create option parameters
                params = OptionParameters(
                    spot_price=spot_price,
                    strike_price=strike_price,
                    time_to_expiry=time_to_expiry / 365.25,
                    volatility=volatility,
                    risk_free_rate=risk_free_rate,
                    option_type=OptionType.CALL if option_type == "Call" else OptionType.PUT
                )
                
                # Initialize PnL simulator
                pnl_sim = TaylorExpansionPnL()
                
                # Calculate PnL components
                pnl_components = pnl_sim.calculate_pnl_components(
                    params, spot_shock=spot_shock, vol_shock=vol_shock, time_decay_days=time_decay
                )
                
                # Display results
                st.success("✅ **PnL Analysis Complete!**")
                
                # Create results columns
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    st.subheader("📊 PnL Breakdown")
                    
                    # Enhanced PnL components table
                    pnl_data = {
                        "Component": ["δΔS (Delta)", "½γ(ΔS)² (Gamma)", "θΔt (Theta)", "νΔσ (Vega)", "**Total PnL**"],
                        "Value ($)": [
                            f"${pnl_components.delta_pnl:.2f}",
                            f"${pnl_components.gamma_pnl:.2f}",
                            f"${pnl_components.theta_pnl:.2f}",
                            f"${pnl_components.vega_pnl:.2f}",
                            f"**${pnl_components.total_pnl:.2f}**"
                        ],
                        "Contribution (%)": [
                            f"{(pnl_components.delta_pnl/pnl_components.total_pnl*100) if pnl_components.total_pnl != 0 else 0:.1f}%",
                            f"{(pnl_components.gamma_pnl/pnl_components.total_pnl*100) if pnl_components.total_pnl != 0 else 0:.1f}%",
                            f"{(pnl_components.theta_pnl/pnl_components.total_pnl*100) if pnl_components.total_pnl != 0 else 0:.1f}%",
                            f"{(pnl_components.vega_pnl/pnl_components.total_pnl*100) if pnl_components.total_pnl != 0 else 0:.1f}%",
                            "100.0%"
                        ]
                    }
                    
                    st.dataframe(pd.DataFrame(pnl_data), hide_index=True)
                    
                    # Enhanced key metrics
                    st.metric("Original Option Price", f"${pnl_components.original_price:.2f}")
                    st.metric("New Theoretical Price", f"${pnl_components.new_theoretical_price:.2f}")
                    st.metric("PnL Change", f"${pnl_components.total_pnl:.2f}", 
                             f"{(pnl_components.total_pnl/pnl_components.original_price*100) if pnl_components.original_price > 0 else 0:.2f}%")
                
                with res_col2:
                    st.subheader("📈 Visual Breakdown")
                    
                    # Enhanced PnL components chart
                    components = ["Delta", "Gamma", "Theta", "Vega"]
                    values = [pnl_components.delta_pnl, pnl_components.gamma_pnl,
                             pnl_components.theta_pnl, pnl_components.vega_pnl]
                    colors = ["blue", "green", "red", "orange"]
                    
                    fig = go.Figure(data=[
                        go.Bar(x=components, y=values, marker_color=colors,
                              text=[f"${v:.2f}" for v in values], textposition='auto')
                    ])
                    
                    fig.update_layout(
                        title="PnL Components Breakdown",
                        xaxis_title="Greeks Components",
                        yaxis_title="PnL ($)",
                        showlegend=False,
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Formula display
                st.subheader("📐 Taylor Expansion Formula")
                st.latex(r"\Delta C \approx \delta \Delta S + \frac{1}{2}\gamma (\Delta S)^2 + \theta \Delta t + \nu \Delta \sigma")
                
                # Enhanced interpretation
                st.subheader("💡 Interpretation")
                interpretation = []
                if pnl_components.delta_pnl > 0:
                    interpretation.append(f"✅ **Delta gain:** ${pnl_components.delta_pnl:.2f} from favorable {selected_currency} price movement")
                else:
                    interpretation.append(f"❌ **Delta loss:** ${pnl_components.delta_pnl:.2f} from unfavorable {selected_currency} price movement")
                
                if pnl_components.gamma_pnl > 0:
                    interpretation.append(f"✅ **Gamma gain:** ${pnl_components.gamma_pnl:.2f} from convexity benefit")
                else:
                    interpretation.append(f"❌ **Gamma loss:** ${pnl_components.gamma_pnl:.2f} from convexity drag")
                
                if pnl_components.theta_pnl < 0:
                    interpretation.append(f"⏰ **Time decay:** ${pnl_components.theta_pnl:.2f} from {time_decay} day(s) passing")
                
                if pnl_components.vega_pnl > 0:
                    interpretation.append(f"📈 **Volatility gain:** ${pnl_components.vega_pnl:.2f} from volatility increase")
                else:
                    interpretation.append(f"📉 **Volatility loss:** ${pnl_components.vega_pnl:.2f} from volatility decrease")
                
                for interp in interpretation:
                    st.markdown(interp)
                
                # Advanced visualization option
                if show_advanced:
                    st.subheader("🎨 Advanced Visualization Options")
                    viz_col1, viz_col2 = st.columns(2)
                    
                    with viz_col1:
                        if st.button("🌊 Generate PnL Heatmap"):
                            st.info("🔗 Navigate to **📊 Advanced Visualizations > 🌊 Taylor PnL Heatmaps** for detailed heatmap analysis")
                    
                    with viz_col2:
                        if st.button("📊 Greeks Dashboard"):
                            st.info("🔗 Navigate to **📊 Advanced Visualizations > 📊 Greeks Risk Dashboard** for comprehensive risk monitoring")
                
        except Exception as e:
            st.error(f"❌ **Calculation failed:** {str(e)}")
            with st.expander("🔍 Error Details"):
                st.code(traceback.format_exc())

# ==========================================
# PLACEHOLDER PAGES (To be implemented)
# ==========================================

def create_volatility_surface_page():
    """Placeholder for volatility surface visualization."""
    st.header("🌋 Volatility Surface")
    st.info("🚧 **Coming Soon:** 3D volatility surface visualization from options_claude.py")
    st.markdown("**Features will include:**")
    st.markdown("- Interactive 3D IV surface across strike/maturity")
    st.markdown("- Real-time parameter adjustment")
    st.markdown("- Export capabilities")

def create_iv_skew_page():
    """Placeholder for IV skew analysis."""
    st.header("📊 IV Skew Analysis")
    st.info("🚧 **Coming Soon:** Enhanced IV skew analysis by maturity")

def create_iv_timeseries_page():
    """Placeholder for IV timeseries."""
    st.header("📈 IV Timeseries")
    st.info("🚧 **Coming Soon:** Volume-weighted IV evolution over time")

def create_distributions_page():
    """Placeholder for option distributions."""
    st.header("📊 Option Distributions")
    st.info("🚧 **Coming Soon:** Strike/volume/maturity distribution analysis")

def create_market_intelligence_page():
    """Placeholder for market intelligence."""
    st.header("📊 Market Intelligence")
    st.info("🚧 **Coming Soon:** Advanced market analysis tools")

def create_greeks_3d_page():
    """Placeholder for Greeks 3D analysis."""
    st.header("⚡ Greeks 3D Analysis")
    st.info("🚧 **Coming Soon:** 3D Greeks surfaces and analysis")

def create_portfolio_greeks_page():
    """Placeholder for portfolio Greeks."""
    st.header("📊 Portfolio Greeks")
    st.info("🚧 **Coming Soon:** Multi-position Greeks aggregation")

def create_risk_monitoring_page():
    """Placeholder for risk monitoring."""
    st.header("🚨 Risk Monitoring")
    st.info("🚧 **Coming Soon:** Real-time risk alerts and monitoring")

def create_sensitivity_analysis_page():
    """Placeholder for sensitivity analysis."""
    st.header("🎯 Sensitivity Analysis")
    st.info("🚧 **Coming Soon:** Parameter sensitivity analysis")

def create_strategy_analysis_page():
    """Placeholder for strategy analysis."""
    st.header("📊 Strategy Analysis")
    st.info("🚧 **Coming Soon:** Multi-strategy payoff analysis")

def create_testing_tools_page():
    """Placeholder for testing tools."""
    st.header("🧪 Testing Tools")
    st.info("🚧 **Coming Soon:** Comprehensive testing suite interface")

# ==========================================
# EXISTING PAGES (Enhanced)
# ==========================================

def create_scenario_analysis_page():
    """Enhanced scenario analysis page."""
    from src.analytics.pnl_simulator import TaylorExpansionPnL, ScenarioParameters
    from src.models.black_scholes import OptionParameters, OptionType
    
    st.header("📈 Scenario Analysis")
    st.markdown("**Comprehensive stress testing with multiple scenarios using real market data**")
    
    # Currency selection
    selected_currency = st.selectbox("Currency", ["BTC", "ETH"], key="scenario_currency")
    smart_defaults = get_smart_defaults(selected_currency)
    display_market_data_status(selected_currency)
    
    # Parameters with smart defaults
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Base Option")
        spot = st.number_input(f"{selected_currency} Spot Price", value=smart_defaults['spot_price'], key="scenario_spot")
        strike = st.number_input("Strike Price", value=smart_defaults['strike_price'], key="scenario_strike")
        tte = st.number_input("Time to Expiry (days)", value=smart_defaults['time_to_expiry_days'], key="scenario_tte") / 365.25
        vol = st.slider("Volatility (%)", 1.0, 200.0, smart_defaults['volatility_percent'], key="scenario_vol") / 100.0
        opt_type = st.selectbox("Option Type", ["Call", "Put"], key="scenario_type")
    
    with col2:
        st.subheader("⚡ Scenario Ranges")
        spot_range = st.slider("Spot Shock Range (%)", 1, 50, 20, key="spot_range")
        vol_range = st.slider("Vol Shock Range (%)", 1, 100, 30, key="vol_range")
        time_max = st.slider("Max Time Decay (days)", 1, 30, 7, key="time_range")
    
    st.info(f"💡 **Using real market data:** Current {selected_currency} price ${smart_defaults['spot_price']:,.0f}, nearest strike ${smart_defaults['strike_price']:,.0f}")
    
    if st.button("🚀 Run Scenario Analysis", key="run_scenarios", type="primary"):
        try:
            with st.spinner("Running comprehensive scenario analysis..."):
                # Create parameters
                params = OptionParameters(
                    spot_price=spot, strike_price=strike, time_to_expiry=tte,
                    volatility=vol, risk_free_rate=0.05,
                    option_type=OptionType.CALL if opt_type == "Call" else OptionType.PUT
                )
                
                # Create scenarios
                scenarios = ScenarioParameters(
                    spot_shocks=[-spot_range/100, -spot_range/200, 0, spot_range/200, spot_range/100],
                    vol_shocks=[-vol_range/100, -vol_range/200, 0, vol_range/200, vol_range/100],
                    time_decay_days=[0, time_max/2, time_max]
                )
                
                # Run analysis
                pnl_sim = TaylorExpansionPnL()
                results = pnl_sim.analyze_scenarios(params, scenarios)
                summary_df = pnl_sim.create_scenario_summary(results)
                risk_metrics = pnl_sim.calculate_risk_metrics(results)
                
                st.success(f"✅ **Analysis Complete! Generated {len(results)} scenarios**")
                
                # Enhanced risk metrics display
                col_a, col_b, col_c, col_d = st.columns(4)
                
                with col_a:
                    st.metric("Mean PnL", f"${risk_metrics['mean_pnl']:.2f}")
                    st.metric("Std Dev", f"${risk_metrics['std_pnl']:.2f}")
                
                with col_b:
                    st.metric("95% VaR", f"${risk_metrics['var_95_pnl']:.2f}")
                    st.metric("99% VaR", f"${risk_metrics['var_99_pnl']:.2f}")

                with col_c:
                    # Enhanced: Show CVaR if available
                    if 'cvar_95_pnl' in risk_metrics:
                        st.metric("95% CVaR", f"${risk_metrics['cvar_95_pnl']:.2f}")
                    if 'cvar_99_pnl' in risk_metrics:
                        st.metric("99% CVaR", f"${risk_metrics['cvar_99_pnl']:.2f}")
                
                with col_d:
                    st.metric("Max Gain", f"${risk_metrics['max_pnl']:.2f}")
                    st.metric("Max Loss", f"${risk_metrics['min_pnl']:.2f}")
                    st.metric("Prob Loss", f"{risk_metrics['prob_loss']:.1f}%")
                
                # Enhanced scenario results
                st.subheader("📋 Detailed Scenario Results")
                
                # Add filters
                filter_col1, filter_col2 = st.columns(2)
                with filter_col1:
                    show_top_n = st.selectbox("Show Top/Bottom Scenarios", [10, 20, 50, "All"])
                with filter_col2:
                    sort_by = st.selectbox("Sort By", ["total_pnl", "delta_pnl", "gamma_pnl", "vega_pnl"])
                
                # Filter and display data
                if show_top_n != "All":
                    display_df = summary_df.head(show_top_n)
                else:
                    display_df = summary_df
                
                st.dataframe(display_df[['scenario_id', 'total_pnl', 'delta_pnl', 'gamma_pnl', 'theta_pnl', 'vega_pnl', 'pnl_percentage']])
                
                # Enhanced PnL distribution chart
                st.subheader("📊 Enhanced PnL Distribution")
                
                fig = px.histogram(summary_df, x='total_pnl', nbins=25, 
                                 title="PnL Distribution Across All Scenarios",
                                 labels={'total_pnl': 'Total PnL ($)', 'count': 'Frequency'})
                
                # Add VaR lines
                fig.add_vline(x=risk_metrics['var_95_pnl'], line_dash="dash", 
                             line_color="orange", annotation_text="95% VaR")
                fig.add_vline(x=risk_metrics['var_99_pnl'], line_dash="dash", 
                             line_color="red", annotation_text="99% VaR")
                
                # Add mean line
                fig.add_vline(x=risk_metrics['mean_pnl'], line_dash="dot", 
                             line_color="green", annotation_text="Mean")
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Download option
                csv = summary_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Scenario Results",
                    data=csv,
                    file_name=f"{selected_currency}_scenario_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"❌ **Analysis failed:** {str(e)}")
            with st.expander("🔍 Error Details"):
                st.code(traceback.format_exc())

def create_greeks_calculator_page():
    """Enhanced Greeks calculation page."""
    from src.models.black_scholes import BlackScholesModel, OptionParameters, OptionType
    
    st.header("🧮 Greeks Calculator")
    st.markdown("**Calculate Black-Scholes Greeks for individual options with real market data**")
    
    # Currency selection
    selected_currency = st.selectbox("Currency", ["BTC", "ETH"], key="greeks_currency")
    smart_defaults = get_smart_defaults(selected_currency)
    display_market_data_status(selected_currency)
    
    # Input parameters with smart defaults
    col1, col2 = st.columns(2)
    
    with col1:
        spot = st.number_input(f"{selected_currency} Spot Price ($)", 
                             value=smart_defaults['spot_price'], min_value=1.0)
        strike = st.number_input("Strike Price ($)", 
                               value=smart_defaults['strike_price'], min_value=1.0)
        time_to_exp = st.number_input("Time to Expiry (days)", 
                                    value=smart_defaults['time_to_expiry_days'], min_value=0.1) / 365.25
    
    with col2:
        vol = st.slider("Volatility (%)", 1.0, 200.0, smart_defaults['volatility_percent']) / 100.0
        rate = st.slider("Risk-Free Rate (%)", 0.0, 20.0, smart_defaults['risk_free_rate_percent']) / 100.0
        opt_type = st.selectbox("Option Type", ["Call", "Put"])
    
    st.info(f"💡 **Using real market data:** Current {selected_currency} price ${smart_defaults['spot_price']:,.0f}")
    
    if st.button("🧮 Calculate Greeks", type="primary"):
        try:
            bs_model = BlackScholesModel()
            
            params = OptionParameters(
                spot_price=spot, strike_price=strike, time_to_expiry=time_to_exp,
                volatility=vol, risk_free_rate=rate,
                option_type=OptionType.CALL if opt_type == "Call" else OptionType.PUT
            )
            
            greeks = bs_model.calculate_greeks(params)
            
            st.success("✅ **Greeks Calculated!**")
            
            # Enhanced results display
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.metric("Option Price", f"${greeks.option_price:.2f}")
                st.metric("Delta", f"{greeks.delta:.4f}")
            
            with col_b:
                st.metric("Gamma", f"{greeks.gamma:.6f}")
                st.metric("Theta", f"${greeks.theta:.4f} (daily)")
            
            with col_c:
                st.metric("Vega", f"${greeks.vega:.4f} (per 1% vol)")
                st.metric("Rho", f"${greeks.rho:.4f} (per 1% rate)")
            
            # Enhanced Greeks interpretation
            st.subheader("💡 Enhanced Greeks Interpretation")
            
            # Moneyness analysis
            moneyness = spot / strike
            if moneyness > 1.05:
                moneyness_desc = "**In-the-Money (ITM)**"
            elif moneyness < 0.95:
                moneyness_desc = "**Out-of-the-Money (OTM)**"
            else:
                moneyness_desc = "**At-the-Money (ATM)**"
            
            st.markdown(f"**Moneyness:** {moneyness:.4f} - {moneyness_desc}")
            
            # Detailed interpretations
            st.markdown(f"- **Delta ({greeks.delta:.4f}):** For every $1 move in {selected_currency}, option price changes by ${greeks.delta:.4f}")
            st.markdown(f"- **Gamma ({greeks.gamma:.6f}):** Delta changes by {greeks.gamma:.6f} for every $1 move in underlying")
            st.markdown(f"- **Theta (${greeks.theta:.4f}):** Option loses ${abs(greeks.theta):.4f} value per day (time decay)")
            st.markdown(f"- **Vega (${greeks.vega:.4f}):** Option price changes by ${greeks.vega:.4f} for 1% volatility change")
            
            # Risk assessment
            st.subheader("⚠️ Risk Assessment")
            
            risk_level = "Low"
            risk_color = "green"
            
            if abs(greeks.delta) > 0.7 or greeks.gamma > 0.1 or abs(greeks.theta) > 50:
                risk_level = "High"
                risk_color = "red"
            elif abs(greeks.delta) > 0.4 or greeks.gamma > 0.05 or abs(greeks.theta) > 25:
                risk_level = "Medium"
                risk_color = "orange"
            
            st.markdown(f"**Overall Risk Level:** :{risk_color}[{risk_level}]")
            
        except Exception as e:
            st.error(f"❌ **Calculation failed:** {str(e)}")
            with st.expander("🔍 Error Details"):
                st.code(traceback.format_exc())

def create_data_collection_page():
    """Enhanced data collection testing page."""
    from src.data.collectors import DeribitCollector
    
    st.header("📊 Data Collection")
    st.markdown("**Test the data collection system and fetch real options data**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌐 API Connection Test")
        if st.button("Test Deribit API Connection"):
            with st.spinner("Testing connection..."):
                try:
                    collector = DeribitCollector()
                    success = collector.test_connection()
                    
                    if success:
                        st.success("✅ **API Connection Successful!**")
                        st.info("🎯 Ready to collect options data")
                    else:
                        st.error("❌ **API Connection Failed**")
                        st.warning("Check internet connection and API status")
                        
                except Exception as e:
                    st.error(f"❌ **Connection Error:** {str(e)}")
    
    with col2:
        st.subheader("📅 Data Collection")
        
        st.info("💡 **Tip:** Recent dates may have limited data. Try historical dates like end of 2024 for better results.")
        
        # Smart date defaults
        suggested_end = date(2024, 12, 31)
        suggested_start = date(2024, 12, 30)
        
        col_a, col_b = st.columns(2)
        with col_a:
            start_date = st.date_input("Start Date", value=suggested_start)
        with col_b:
            end_date = st.date_input("End Date", value=suggested_end)
        
        currency = st.selectbox("Currency", ["BTC", "ETH"])
        timeout_seconds = st.selectbox("Timeout (seconds)", [30, 60, 120], index=1)
        
        if st.button("🚀 Collect Options Data", type="primary"):
            if start_date <= end_date:
                progress_container = st.container()
                status_container = st.container()
                result_container = st.container()
                
                with progress_container:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                
                with status_container:
                    status_text.text("🔄 Initializing data collection...")
                
                try:
                    progress_bar.progress(10)
                    status_text.text("🔗 Connecting to Deribit API...")
                    
                    collector = DeribitCollector()
                    
                    progress_bar.progress(20)
                    status_text.text("✅ API Connected - Testing...")
                    
                    # Test connection first
                    if not collector.test_connection():
                        st.error("❌ **API connection test failed**")
                        return
                    
                    progress_bar.progress(40)
                    status_text.text(f"📊 Collecting {currency} options data...")
                    
                    # Collect data with timeout
                    data = collector.collect_options_data(
                        currency=currency, start_date=start_date, end_date=end_date,
                        max_collection_time=timeout_seconds
                    )
                    
                    progress_bar.progress(90)
                    status_text.text("🔄 Processing collected data...")
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Data collection completed!")
                    
                    # Display results
                    with result_container:
                        if data is not None and not data.empty:
                            st.success(f"✅ **Collected {len(data)} option records!**")
                            
                            # Enhanced data display
                            tab1, tab2, tab3 = st.tabs(["📋 Sample Data", "📊 Summary", "📈 Analysis"])
                            
                            with tab1:
                                st.dataframe(data.head(10))
                            
                            with tab2:
                                col_i, col_ii, col_iii = st.columns(3)
                                
                                with col_i:
                                    st.metric("Total Records", len(data))
                                    if 'strike_price' in data.columns:
                                        st.metric("Unique Strikes", data['strike_price'].nunique())
                                
                                with col_ii:
                                    if 'option_type' in data.columns:
                                        calls = len(data[data['option_type'].str.lower().str.contains('c', na=False)])
                                        puts = len(data) - calls
                                        st.metric("Call Options", calls)
                                        st.metric("Put Options", puts)
                                
                                with col_iii:
                                    if 'implied_volatility' in data.columns:
                                        st.metric("Avg IV", f"{data['implied_volatility'].mean():.2%}")
                                    price_col = next((col for col in ['underlying_price', 'index_price'] if col in data.columns), None)
                                    if price_col:
                                        st.metric("Price Range", f"${data[price_col].min():.0f} - ${data[price_col].max():.0f}")
                            
                            with tab3:
                                if 'implied_volatility' in data.columns and 'strike_price' in data.columns:
                                    fig = px.scatter(data, x='strike_price', y='implied_volatility', 
                                                   color='option_type', title="IV vs Strike Price")
                                    st.plotly_chart(fig, use_container_width=True)
                            
                            # Download option
                            csv = data.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Data as CSV",
                                data=csv,
                                file_name=f"{currency}_options_{start_date}_{end_date}.csv",
                                mime="text/csv"
                            )
                            
                        else:
                            st.warning("⚠️ **No data collected**")
                            st.info("💡 **Try these solutions:**")
                            st.info("- Use historical dates (end of 2024 works well)")
                            st.info("- Try a wider date range (7-30 days)")
                            st.info("- Check if the currency has active options trading")
                    
                except Exception as e:
                    st.error(f"❌ **Data collection failed:** {str(e)}")
                    with st.expander("🔍 Technical Details"):
                        st.code(traceback.format_exc())
                        
            else:
                st.error("❌ Start date must be before end date")

def create_system_status_page():
    """Enhanced system status and information page."""
    st.header("ℹ️ System Status")
    
    # Test all backend components
    st.subheader("🔧 Backend Component Status")
    
    components = {
        "Taylor Expansion PnL": "src.analytics.pnl_simulator",
        "Black-Scholes Model": "src.models.black_scholes", 
        "Data Collectors": "src.data.collectors",
        "Time Utilities": "src.utils.time_utils",
        "Asset Configuration": "src.config.assets",
        "Enhanced Visualizations": "src.visualization.enhanced_viz_framework"  # NEW
    }
    
    for name, module in components.items():
        try:
            __import__(module)
            st.success(f"✅ **{name}** - Working")
        except Exception as e:
            st.error(f"❌ **{name}** - Error: {str(e)}")
    
    # Test market data
    st.subheader("📊 Market Data Status")
    try:
        btc_price = get_current_crypto_price("BTC")
        eth_price = get_current_crypto_price("ETH")
        st.success(f"✅ **Market Data API** - Working")
        st.info(f"Current prices: BTC ${btc_price:,.2f}, ETH ${eth_price:,.2f}")
    except Exception as e:
        st.error(f"❌ **Market Data API** - Error: {str(e)}")
    
    # Enhanced package info
    st.subheader("📦 Enhanced Package Information")
    try:
        from src import get_package_info
        info = get_package_info()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Version:** {info['version']}")
            st.markdown(f"**Python:** {info['python_version']}")
            st.markdown(f"**Core Dependencies:** {info['core_deps_available']}")
        
        with col2:
            st.markdown(f"**All Systems Ready:** {'✅' if info['all_systems_ready'] else '❌'}")
            st.markdown(f"**Advanced Finance:** {'✅' if info['advanced_finance_ready'] else '❌'}")
            st.markdown(f"**Continuous Collector:** {'✅' if info['continuous_collector_ready'] else '❌'}")
            
    except Exception as e:
        st.error(f"Could not load package info: {e}")
    
    # Enhanced feature status
    st.subheader("🎯 Enhanced Feature Implementation Status")
    
    features = {
        "Primary Feature (Taylor Expansion PnL)": "✅ COMPLETE",
        "Black-Scholes Greeks": "✅ COMPLETE", 
        "Data Collection System": "✅ COMPLETE",
        "Smart Market Data Defaults": "✅ COMPLETE",
        "Asset Discovery": "✅ COMPLETE",
        "Enhanced Visualization Framework": "✅ NEW - INTEGRATED",  # NEW
        "Taylor PnL Heatmaps": "✅ NEW - AVAILABLE",  # NEW
        "Greeks Risk Dashboard": "✅ NEW - AVAILABLE",  # NEW
        "PnL Components Analysis": "✅ NEW - AVAILABLE",  # NEW
        "Interactive Parameter Explorer": "✅ NEW - AVAILABLE",  # NEW
        "Dashboard Interface": "✅ ENHANCED",
        "CLI Interface": "🔄 IN PROGRESS",
        "Continuous Data Collection": "✅ AVAILABLE",
        "Risk Management Tools": "✅ ENHANCED"
    }
    
    for feature, status in features.items():
        if "✅" in status:
            if "NEW" in status:
                st.success(f"🆕 **{feature}:** {status}")
            else:
                st.success(f"**{feature}:** {status}")
        elif "🔄" in status:
            st.warning(f"**{feature}:** {status}")
        else:
            st.info(f"**{feature}:** {status}")
    
    # NEW: Visualization capabilities
    st.subheader("🎨 Available Visualization Types")
    
    viz_types = [
        "🌊 Taylor Expansion PnL Heatmaps",
        "📊 Greeks Risk Dashboard", 
        "📈 PnL Components Analysis",
        "🎛️ Interactive Parameter Explorer",
        "🌋 Volatility Surface (Coming Soon)",
        "📊 IV Skew Analysis (Coming Soon)",
        "📈 IV Timeseries (Coming Soon)",
        "📊 Option Distributions (Coming Soon)"
    ]
    
    for viz_type in viz_types:
        if "Coming Soon" in viz_type:
            st.info(f"🚧 {viz_type}")
        else:
            st.success(f"✅ {viz_type}")

def main():
    """Main dashboard entry point."""
    if not STREAMLIT_AVAILABLE:
        print("❌ Streamlit not available!")
        print("Install with: pip install streamlit")
        return
    
    try:
        create_enhanced_dashboard()
        logger.info("Enhanced dashboard initialized successfully")
    except Exception as e:
        logger.error(f"Dashboard initialization failed: {e}")
        st.error(f"Dashboard Error: {e}")
        st.error("Full traceback:")
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
