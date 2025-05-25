"""
Bitcoin Options Analytics Platform - Streamlit Dashboard

This is the main entry point for the interactive Streamlit dashboard.
It provides a web-based interface for Bitcoin options analysis and PnL simulation.

Usage:
    streamlit run dashboard/app.py
    python -c "from dashboard.app import main; main()"
"""

import sys
import logging
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_placeholder_dashboard():
    """Create a placeholder dashboard when Streamlit is available."""
    if not STREAMLIT_AVAILABLE:
        return
    
    try:
        from src import __version__
    except ImportError:
        __version__ = "1.0.0"
    
    # Page configuration
    st.set_page_config(
        page_title="Bitcoin Options Analytics",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Main header
    st.title("🚀 Bitcoin Options Analytics Platform")
    st.markdown(f"**Version:** {__version__}")
    
    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📈 Options Analysis Dashboard")
        
        # Placeholder content
        st.info("🚧 **Dashboard Under Construction**")
        st.markdown("""
        This interactive dashboard will provide:
        
        **✅ Real-time Options Data**
        - Live BTC options prices from Deribit
        - Implied volatility surfaces
        - Greeks calculations
        
        **✅ Taylor Expansion PnL Analysis**
        - Interactive scenario analysis: `ΔC ≈ δΔS + ½γ(ΔS)² + θΔt + νΔσ`
        - Stress testing with custom parameters
        - Risk metrics and portfolio analysis
        
        **✅ Interactive Visualizations**
        - Volatility surface plots
        - PnL heatmaps
        - Time series analysis
        - Greeks monitoring
        """)
        
        # Sample interactive elements
        st.subheader("🎛️ Analysis Parameters")
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            spot_price = st.number_input(
                "BTC Spot Price ($)",
                min_value=10000,
                max_value=200000,
                value=30000,
                step=1000
            )
        
        with col_b:
            strike_price = st.number_input(
                "Strike Price ($)",
                min_value=10000,
                max_value=200000,
                value=32000,
                step=1000
            )
        
        with col_c:
            volatility = st.slider(
                "Implied Volatility",
                min_value=0.1,
                max_value=2.0,
                value=0.8,
                step=0.05
            )
        
        # Calculate basic metrics (placeholder)
        moneyness = spot_price / strike_price
        st.metric("Moneyness (S/K)", f"{moneyness:.3f}")
        
        if st.button("🔄 Run Analysis", type="primary"):
            st.success("✅ Analysis would run here!")
            st.balloons()
    
    with col2:
        st.header("⚙️ System Status")
        
        # System status indicators
        st.metric("📡 API Status", "🟡 Not Connected")
        st.metric("💾 Data Cache", "🟢 Ready")
        st.metric("🧮 Models", "🟢 Loaded")
        
        st.subheader("📋 Quick Actions")
        
        if st.button("📊 Collect Data"):
            st.info("Data collection module will be implemented")
        
        if st.button("🧮 Calculate Greeks"):
            st.info("Greeks calculation module will be implemented")
        
        if st.button("📈 PnL Simulation"):
            st.info("PnL simulation module will be implemented")
        
        # Recent activity placeholder
        st.subheader("📈 Recent Activity")
        st.text("• Dashboard initialized")
        st.text("• Waiting for data connection")
        st.text("• Ready for analysis")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "🔗 **Bitcoin Options Analytics Platform** • "
        "Built with Streamlit • "
        f"v{__version__}"
    )

def main():
    """Main dashboard entry point."""
    if not STREAMLIT_AVAILABLE:
        print("❌ Streamlit not available!")
        print("Install with: pip install streamlit")
        print("Or use conda: conda install -c conda-forge streamlit")
        return
    
    try:
        create_placeholder_dashboard()
        logger.info("Dashboard initialized successfully")
    except Exception as e:
        logger.error(f"Dashboard initialization failed: {e}")
        if STREAMLIT_AVAILABLE:
            st.error(f"Dashboard Error: {e}")

# For running with streamlit command
if __name__ == "__main__":
    main()

# Alternative entry point for GUI scripts
def streamlit_main():
    """Alternative entry point for setup.py gui_scripts."""
    if not STREAMLIT_AVAILABLE:
        print("❌ Error: Streamlit is required but not installed")
        print("Install with: pip install streamlit")
        sys.exit(1)
    
    import subprocess
    import os
    
    # Get the path to this file
    dashboard_path = Path(__file__).absolute()
    
    # Run streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(dashboard_path)
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start dashboard: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Dashboard stopped by user")
        sys.exit(0)
