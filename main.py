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
        # This will be implemented when we create the data collection module
        print("⚠️  Data collection module not yet implemented")
        print("✅ This will collect options data from Deribit API")
        print("✅ Process and analyze the data")
        print("✅ Generate analysis reports")
        
        # Placeholder for actual implementation
        logger.info("Options analysis completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return False

def start_dashboard(args):
    """Start the Streamlit dashboard."""
    print("🚀 Starting interactive dashboard...")
    
    try:
        # This will be implemented when we create the dashboard
        print("⚠️  Dashboard module not yet implemented")
        print("✅ This will start Streamlit on http://localhost:8501")
        print("✅ Interactive PnL analysis interface")
        print("✅ Real-time options data visualization")
        
        # Placeholder for actual implementation
        logger.info("Dashboard started successfully")
        return True
        
    except Exception as e:
        logger.error(f"Dashboard startup failed: {e}")
        return False

def simulate_pnl(args):
    """Run PnL simulation."""
    print("📊 Running Taylor expansion PnL simulation...")
    
    try:
        # This will be implemented when we create the PnL module
        print("⚠️  PnL simulation module not yet implemented")
        print("✅ This will run: ΔC ≈ δΔS + ½γ(ΔS)² + θΔt + νΔσ")
        print("✅ Generate scenario analysis")
        print("✅ Calculate risk metrics")
        
        # Placeholder for actual implementation
        logger.info("PnL simulation completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"PnL simulation failed: {e}")
        return False

def show_info(args):
    """Show package information."""
    info = get_package_info()
    print("\n📋 Package Information:")
    for key, value in info.items():
        print(f"   {key.replace('_', ' ').title()}: {value}")
    print()

def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description="Bitcoin Options Analytics Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py info                          # Show package info
  python main.py analyze --currency BTC       # Analyze BTC options
  python main.py dashboard                     # Start web dashboard
  python main.py simulate-pnl                  # Run PnL simulation
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
        default=date.today(),
        help="Start date (YYYY-MM-DD)"
    )
    analyze_parser.add_argument(
        "--end-date",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        default=date.today(),
        help="End date (YYYY-MM-DD)"
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
        default=100,
        help="Number of scenarios to simulate (default: 100)"
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
