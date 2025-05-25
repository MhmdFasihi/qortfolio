"""
Command Line Interface for Bitcoin Options Analytics Platform

This module provides CLI functions that are used as entry points in setup.py.
It includes specialized CLI interfaces for different components of the system.
"""

import sys
import click
import logging
from typing import Optional
from datetime import date, datetime

from . import __version__, configure_logging

# Configure logging
logger = logging.getLogger(__name__)

# Click group for main CLI
@click.group()
@click.version_option(version=__version__)
@click.option('--log-level', 
              type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']),
              default='INFO',
              help='Set logging level')
@click.pass_context
def cli(ctx, log_level):
    """Bitcoin Options Analytics Platform CLI."""
    configure_logging(log_level)
    ctx.ensure_object(dict)

@cli.command()
def info():
    """Show package information."""
    from . import get_package_info
    
    click.echo("📋 Bitcoin Options Analytics Platform")
    click.echo("=" * 40)
    
    info = get_package_info()
    for key, value in info.items():
        click.echo(f"  {key.replace('_', ' ').title()}: {value}")

@cli.command()
@click.option('--currency', default='BTC', help='Currency to analyze')
@click.option('--start-date', type=click.DateTime(formats=['%Y-%m-%d']),
              default=str(date.today()), help='Start date (YYYY-MM-DD)')
@click.option('--end-date', type=click.DateTime(formats=['%Y-%m-%d']),
              default=str(date.today()), help='End date (YYYY-MM-DD)')
def analyze(currency, start_date, end_date):
    """Analyze options data."""
    click.echo(f"🔍 Analyzing {currency} options...")
    click.echo(f"📅 Date range: {start_date.date()} to {end_date.date()}")
    
    # Placeholder for actual implementation
    click.echo("⚠️  Data collection module not yet implemented")
    click.echo("✅ This will collect and analyze options data")
    
    logger.info(f"Analysis command executed for {currency}")

@cli.command()
@click.option('--port', default=8501, help='Port for dashboard')
def dashboard(port):
    """Start interactive dashboard."""
    click.echo("🚀 Starting interactive dashboard...")
    click.echo(f"📊 Dashboard will be available at http://localhost:{port}")
    
    # Placeholder for actual implementation
    click.echo("⚠️  Dashboard module not yet implemented")
    click.echo("✅ This will start the Streamlit dashboard")
    
    logger.info(f"Dashboard command executed on port {port}")

@cli.command()
@click.option('--scenarios', default=100, help='Number of scenarios to simulate')
def simulate_pnl(scenarios):
    """Run PnL simulation."""
    click.echo("📊 Running Taylor expansion PnL simulation...")
    click.echo(f"🎯 Simulating {scenarios} scenarios")
    
    # Placeholder for actual implementation
    click.echo("⚠️  PnL simulation module not yet implemented")
    click.echo("✅ This will run: ΔC ≈ δΔS + ½γ(ΔS)² + θΔt + νΔσ")
    
    logger.info(f"PnL simulation command executed with {scenarios} scenarios")

def main():
    """Main CLI entry point for setup.py console_scripts."""
    try:
        cli()
    except KeyboardInterrupt:
        click.echo("\n⚠️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"CLI error: {e}")
        click.echo(f"❌ Error: {e}")
        sys.exit(1)

def analyzer_main():
    """Analyzer-specific CLI entry point."""
    try:
        cli(['analyze'])
    except KeyboardInterrupt:
        click.echo("\n⚠️  Analysis cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Analyzer error: {e}")
        click.echo(f"❌ Analysis error: {e}")
        sys.exit(1)

def pnl_main():
    """PnL simulation CLI entry point."""
    try:
        cli(['simulate-pnl'])
    except KeyboardInterrupt:
        click.echo("\n⚠️  PnL simulation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"PnL simulation error: {e}")
        click.echo(f"❌ PnL simulation error: {e}")
        sys.exit(1)

# Alternative simple CLI functions if Click is not available
def simple_main():
    """Simple main CLI without Click dependency."""
    print(f"🚀 Bitcoin Options Analytics Platform v{__version__}")
    print("=" * 50)
    print("Available commands:")
    print("  • python -m src.cli info          - Show package info")
    print("  • python -m src.cli analyze       - Analyze options")
    print("  • python -m src.cli dashboard     - Start dashboard")
    print("  • python -m src.cli simulate-pnl  - Run PnL simulation")
    print("=" * 50)

if __name__ == "__main__":
    main()
