#!/usr/bin/env python3
"""
Bitcoin Options Analytics Platform - Main Entry Point

This is the main entry point for the Bitcoin Options Analytics Platform.
It provides a command-line interface for running various components of the system.

Usage:
    python main.py --help
    python main.py analyze --currency BTC --start-date 2025-01-20
    python main.py dashboard
    python main.py simulate-pnl
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import date, datetime
from typing import Optional

# Add src to Python path for development
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src import __version__, get_package_info, configure_logging
except ImportError:
    # Fallback for development
    __version__ = "1.0.0"
    
    def get_package_info():
        return {"name": "bitcoin-options-analytics", "version": __version__}
    
    def configure_logging(level="INFO"):
        logging.basicConfig(level=getattr(logging, level))

# Configure logging
logger = logging.getLogger(__name__)

def print_banner():
    """Print application banner."""
    banner = f"""
{'='*60}
🚀 Bitcoin Options Analytics Platform v{__version__}
{'='*60}
Professional Bitcoin options analysis with Taylor expansion PnL
Real-time data • Interactive dashboards • Risk management
{'='*60}
    """
    print(banner)

def analyze_options(args):
    """Run options analysis."""
    print(f"🔍 Analyzing {args.currency} options...")
    print(f"📅 Date range: {args.start_date} to {args.end_date}")
    
    try:
        # Import and use the actual data collector
        from src.data.collectors import DeribitCollector
        
        print("✅ Initializing data collector...")
        with DeribitCollector() as collector:
            print("✅ Testing API connection...")
            if not collector.test_connection():
                print("❌ API connection failed")
                return False
            
            print("✅ Collecting options data...")
            data = collector.collect_options_data(
                currency=args.currency,
                start_date=args.start_date,
                end_date=args.end_date
            )
            
            if data.empty:
                print("⚠️  No data collected for the specified date range")
                print("💡 Try a different date range or check API status")
                return True
            
            print(f"✅ Collected {len(data)} option records")
            
            # Calculate Greeks if requested
            if hasattr(args, 'calculate_greeks') and args.calculate_greeks:
                print("🧮 Calculating Greeks...")
                from src.models.black_scholes import BlackScholesModel
                
                bs_model = BlackScholesModel()
                enhanced_data = bs_model.calculate_greeks_for_dataframe(data)
                
                print(f"✅ Greeks calculated for {len(enhanced_data)} options")
                
                # Save enhanced data
                output_file = f"{args.currency}_options_with_greeks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                enhanced_data.to_csv(output_file, index=False)
                print(f"✅ Enhanced data saved to {output_file}")
                
                return True
            
            # Save basic data
            output_file = f"{args.currency}_options_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            data.to_csv(output_file, index=False)
            print(f"✅ Data saved to {output_file}")
            
        logger.info("Options analysis completed successfully")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all dependencies are installed: pip install -r requirements.txt")
        return False
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"❌ Analysis failed: {e}")
        return False

def start_dashboard(args):
    """Start the Streamlit dashboard."""
    print("🚀 Starting interactive dashboard...")
    
    try:
        import subprocess
        import sys
        
        # Try to import streamlit to check if it's available
        try:
            import streamlit
            print("✅ Streamlit available")
        except ImportError:
            print("❌ Streamlit not installed")
            print("💡 Install with: pip install streamlit")
            return False
        
        # Check if dashboard app exists
        dashboard_path = Path("dashboard/app.py")
        if not dashboard_path.exists():
            print("❌ Dashboard app not found at dashboard/app.py")
            print("💡 Make sure the dashboard module is properly set up")
            return False
        
        print(f"✅ Starting dashboard on port {args.port}...")
        print(f"🌐 Dashboard will be available at http://localhost:{args.port}")
        
        # Start streamlit
        cmd = [sys.executable, "-m", "streamlit", "run", str(dashboard_path), "--server.port", str(args.port)]
        
        print("📊 Launching Streamlit dashboard...")
        print("⚠️  Note: Use Ctrl+C to stop the dashboard")
        
        # Run streamlit
        result = subprocess.run(cmd, check=True)
        
        logger.info(f"Dashboard started successfully on port {args.port}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Dashboard startup failed: {e}")
        logger.error(f"Dashboard startup failed: {e}")
        return False
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Install missing dependencies: pip install streamlit")
        return False
    except Exception as e:
        logger.error(f"Dashboard startup failed: {e}")
        print(f"❌ Unexpected error: {e}")
        return False

def simulate_pnl(args):
    """Run PnL simulation."""
    print("📊 Running Taylor expansion PnL simulation...")
    print(f"🎯 Simulating {args.scenarios} scenarios")
    
    try:
        # Check if we have the required models
        from src.models.black_scholes import BlackScholesModel, OptionParameters, OptionType
        
        print("✅ Black-Scholes model available")
        
        # Example simulation parameters
        print("🔧 Setting up simulation parameters...")
        
        # Create a sample option for demonstration
        params = OptionParameters(
            spot_price=30000,  # Current BTC price
            strike_price=32000,  # Strike price
            time_to_expiry=30/365.25,  # 30 days to expiry
            volatility=0.80,  # 80% implied volatility
            risk_free_rate=0.05,  # 5% risk-free rate
            option_type=OptionType.CALL
        )
        
        print("📈 Sample Option Parameters:")
        print(f"   Spot Price: ${params.spot_price:,.0f}")
        print(f"   Strike Price: ${params.strike_price:,.0f}")
        print(f"   Time to Expiry: {params.time_to_expiry:.4f} years ({params.time_to_expiry*365:.0f} days)")
        print(f"   Volatility: {params.volatility:.1%}")
        print(f"   Risk-free Rate: {params.risk_free_rate:.1%}")
        print(f"   Option Type: {params.option_type.value}")
        
        # Calculate base Greeks
        bs_model = BlackScholesModel()
        greeks = bs_model.calculate_greeks(params)
        
        print("\n🧮 Base Greeks:")
        print(f"   Option Price: ${greeks.option_price:.2f}")
        print(f"   Delta: {greeks.delta:.4f}")
        print(f"   Gamma: {greeks.gamma:.6f}")
        print(f"   Theta: ${greeks.theta:.4f} (daily)")
        print(f"   Vega: ${greeks.vega:.4f} (per 1% vol change)")
        print(f"   Rho: ${greeks.rho:.4f} (per 1% rate change)")
        
        # Simple PnL simulation using Taylor expansion
        print(f"\n📊 Running {args.scenarios} Taylor Expansion PnL scenarios...")
        
        import numpy as np
        
        # Generate random scenarios
        np.random.seed(42)  # For reproducible results
        
        # Scenario parameters
        spot_changes = np.random.normal(0, 0.05, args.scenarios)  # 5% daily volatility
        vol_changes = np.random.normal(0, 0.1, args.scenarios)   # 10% vol volatility
        time_decay = 1/365.25  # 1 day time decay
        
        pnl_results = []
        
        for i in range(args.scenarios):
            # Taylor expansion: ΔC ≈ δΔS + ½γ(ΔS)² + θΔt + νΔσ
            ds = spot_changes[i] * params.spot_price  # Price change
            dsigma = vol_changes[i]  # Vol change
            dt = time_decay  # Time decay
            
            # Calculate PnL components
            delta_pnl = greeks.delta * ds
            gamma_pnl = 0.5 * greeks.gamma * ds**2
            theta_pnl = greeks.theta * dt
            vega_pnl = greeks.vega * dsigma
            
            total_pnl = delta_pnl + gamma_pnl + theta_pnl + vega_pnl
            
            pnl_results.append({
                'scenario': i+1,
                'spot_change': ds,
                'vol_change': dsigma,
                'delta_pnl': delta_pnl,
                'gamma_pnl': gamma_pnl,
                'theta_pnl': theta_pnl,
                'vega_pnl': vega_pnl,
                'total_pnl': total_pnl
            })
        
        # Calculate statistics
        total_pnls = [result['total_pnl'] for result in pnl_results]
        
        print("\n📈 PnL Simulation Results:")
        print(f"   Scenarios: {len(pnl_results)}")
        print(f"   Mean PnL: ${np.mean(total_pnls):.2f}")
        print(f"   Std Dev: ${np.std(total_pnls):.2f}")
        print(f"   Min PnL: ${np.min(total_pnls):.2f}")
        print(f"   Max PnL: ${np.max(total_pnls):.2f}")
        print(f"   95% VaR: ${np.percentile(total_pnls, 5):.2f}")
        
        # Save results if requested
        if hasattr(args, 'save_results') and args.save_results:
            import pandas as pd
            
            df = pd.DataFrame(pnl_results)
            output_file = f"pnl_simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(output_file, index=False)
            print(f"✅ Results saved to {output_file}")
        
        logger.info(f"PnL simulation completed with {args.scenarios} scenarios")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all dependencies are installed")
        return False
    except Exception as e:
        logger.error(f"PnL simulation failed: {e}")
        print(f"❌ Simulation failed: {e}")
        return False

def show_info(args):
    """Show package information."""
    info = get_package_info()
    print("\n📋 Package Information:")
    for key, value in info.items():
        if key == 'dependencies':
            print("   Dependencies:")
            for dep, available in value.items():
                status = "✅" if available else "❌"
                print(f"     {status} {dep}")
        else:
            print(f"   {key.replace('_', ' ').title()}: {value}")
    print()

def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description="Bitcoin Options Analytics Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py info                                    # Show package info
  python main.py analyze --currency BTC                 # Analyze BTC options
  python main.py analyze --currency BTC --calculate-greeks  # With Greeks
  python main.py dashboard                               # Start web dashboard
  python main.py simulate-pnl --scenarios 1000          # Run PnL simulation
        """
    )
    
    # Global arguments
    parser.add_argument(
        "--version", 
        action="version", 
        version=f"Bitcoin Options Analytics Platform v{__version__}"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set logging level (default: INFO)"
    )
    parser.add_argument(
        "--no-banner",
        action="store_true",
        help="Skip banner display"
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show package information")
    info_parser.set_defaults(func=show_info)
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze options data")
    analyze_parser.add_argument(
        "--currency", 
        default="BTC", 
        help="Currency to analyze (default: BTC)"
    )
    analyze_parser.add_argument(
        "--start-date",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        default=date.today() - datetime.timedelta(days=7),
        help="Start date (YYYY-MM-DD, default: 7 days ago)"
    )
    analyze_parser.add_argument(
        "--end-date",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        default=date.today(),
        help="End date (YYYY-MM-DD, default: today)"
    )
    analyze_parser.add_argument(
        "--calculate-greeks",
        action="store_true",
        help="Calculate Greeks for all options"
    )
    analyze_parser.set_defaults(func=analyze_options)
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser("dashboard", help="Start interactive dashboard")
    dashboard_parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port for dashboard (default: 8501)"
    )
    dashboard_parser.set_defaults(func=start_dashboard)
    
    # PnL simulation command
    pnl_parser = subparsers.add_parser("simulate-pnl", help="Run PnL simulation")
    pnl_parser.add_argument(
        "--scenarios",
        type=int,
        default=1000,
        help="Number of scenarios to simulate (default: 1000)"
    )
    pnl_parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save simulation results to CSV"
    )
    pnl_parser.set_defaults(func=simulate_pnl)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Configure logging
    configure_logging(args.log_level)
    
    # Show banner unless disabled
    if not args.no_banner:
        print_banner()
    
    # Handle no command
    if not args.command:
        parser.print_help()
        return 0
    
    # Execute command
    try:
        success = args.func(args)
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())