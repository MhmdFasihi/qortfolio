#!/usr/bin/env python3
"""
# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com
Bitcoin Options Continuous Data Collector - Monitor Script

This script monitors the health and status of the continuous data collector.
It reads status files and provides real-time insights into collector performance.

Usage:
    python monitor_collector.py              # Single status check
    python monitor_collector.py --watch      # Continuous monitoring
    python monitor_collector.py --detailed   # Detailed status report
    python monitor_collector.py --alerts     # Show only alerts/issues
"""

import json
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

def load_collector_status(status_dir: str = "continuous_data") -> Optional[Dict[str, Any]]:
    """
    Load the current collector status from JSON file.
    
    Args:
        status_dir: Directory containing status files
        
    Returns:
        Status dictionary or None if not found
    """
    status_file = Path(status_dir) / "status" / "collector_status.json"
    
    if not status_file.exists():
        return None
    
    try:
        with open(status_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error reading status file: {e}")
        return None

def format_time_ago(timestamp_str: str) -> str:
    """
    Format timestamp as time ago (e.g., '5 minutes ago').
    
    Args:
        timestamp_str: ISO format timestamp string
        
    Returns:
        Human-readable time difference
    """
    try:
        timestamp = datetime.fromisoformat(timestamp_str)
        now = datetime.now()
        
        # Handle timezone-aware timestamps
        if timestamp.tzinfo is not None and now.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=None)
        elif timestamp.tzinfo is None and now.tzinfo is not None:
            now = now.replace(tzinfo=None)
            
        diff = now - timestamp
        
        if diff.days > 0:
            return f"{diff.days} day{'s' if diff.days > 1 else ''} ago"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"{hours} hour{'s' if hours > 1 else ''} ago"
        elif diff.seconds > 60:
            minutes = diff.seconds // 60
            return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
        else:
            return f"{diff.seconds} second{'s' if diff.seconds != 1 else ''} ago"
    except Exception:
        return "unknown"

def get_status_emoji(status: str) -> str:
    """Get emoji for status."""
    emoji_map = {
        'running': '🟢',
        'stopped': '🔴',
        'paused': '🟡',
        'error': '🔴',
        'starting': '🟡',
        'stopping': '🟡'
    }
    return emoji_map.get(status.lower(), '⚪')

def get_health_emoji(health: str) -> str:
    """Get emoji for health status."""
    emoji_map = {
        'healthy': '💚',
        'warning': '⚠️',
        'critical': '🚨',
        'unknown': '❓'
    }
    return emoji_map.get(health.lower(), '❓')

def print_basic_status(status: Dict[str, Any]) -> bool:
    """
    Print basic status information.
    
    Args:
        status: Status dictionary
        
    Returns:
        True if healthy, False if issues detected
    """
    print("🚀 Bitcoin Options Continuous Collector - Status Monitor")
    print("=" * 60)
    
    # Main status
    status_emoji = get_status_emoji(status['status'])
    health_emoji = get_health_emoji(status['health'])
    
    print(f"Status: {status_emoji} {status['status'].upper()}")
    print(f"Health: {health_emoji} {status['health'].upper()}")
    print(f"Uptime: ⏱️  {status['uptime_hours']:.1f} hours")
    print(f"Running: {'✅ YES' if status['is_running'] else '❌ NO'}")
    
    # Last update
    if 'last_updated' in status:
        last_update = format_time_ago(status['last_updated'])
        print(f"Last Update: 🕐 {last_update}")
    
    print()
    
    # Currency summary
    print("📊 CURRENCY SUMMARY:")
    print("-" * 40)
    
    currencies = status.get('currencies', {})
    total_records = 0
    total_failures = 0
    
    for currency, stats in currencies.items():
        success_rate = stats['success_rate']
        records = stats['total_records']
        failures = stats['consecutive_failures']
        
        total_records += records
        total_failures += failures
        
        # Status icon based on performance
        if success_rate >= 90 and failures == 0:
            icon = "✅"
        elif success_rate >= 70 and failures < 3:
            icon = "⚠️"
        else:
            icon = "❌"
        
        print(f"{icon} {currency}:")
        print(f"    Success Rate: {success_rate:.1f}%")
        print(f"    Records: {records:,}")
        print(f"    Consecutive Failures: {failures}")
        
        # Last success
        last_success = stats.get('last_success')
        if last_success:
            last_success_ago = format_time_ago(last_success)
            print(f"    Last Success: {last_success_ago}")
        else:
            print(f"    Last Success: Never")
        print()
    
    # Overall summary
    print("📈 OVERALL SUMMARY:")
    print("-" * 40)
    print(f"Total Records Collected: {total_records:,}")
    print(f"Total Active Failures: {total_failures}")
    
    # Health assessment
    is_healthy = (
        status['health'] == 'healthy' and 
        status['is_running'] and 
        total_failures < 5
    )
    
    if is_healthy:
        print("💚 System Status: HEALTHY")
    elif status['health'] == 'warning':
        print("⚠️  System Status: WARNING - Monitor closely")
    else:
        print("🚨 System Status: NEEDS ATTENTION")
    
    return is_healthy

def print_detailed_status(status: Dict[str, Any]) -> None:
    """Print detailed status information."""
    print_basic_status(status)
    
    print("\n" + "=" * 60)
    print("🔍 DETAILED STATISTICS")
    print("=" * 60)
    
    currencies = status.get('currencies', {})
    
    for currency, stats in currencies.items():
        print(f"\n📊 {currency} DETAILED STATS:")
        print("-" * 30)
        print(f"Total Runs: {stats['total_runs']}")
        print(f"Success Rate: {stats['success_rate']:.1f}%")
        print(f"Records Collected: {stats['total_records']:,}")
        print(f"Consecutive Failures: {stats['consecutive_failures']}")
        
        # Calculate performance metrics
        if stats['total_runs'] > 0:
            avg_records_per_run = stats['total_records'] / stats['total_runs']
            print(f"Avg Records/Run: {avg_records_per_run:.1f}")
        
        # Last success details
        last_success = stats.get('last_success')
        if last_success:
            print(f"Last Success: {last_success}")
            print(f"Time Since Success: {format_time_ago(last_success)}")
        else:
            print("Last Success: Never")

def check_for_alerts(status: Dict[str, Any]) -> list:
    """
    Check for alert conditions.
    
    Args:
        status: Status dictionary
        
    Returns:
        List of alert messages
    """
    alerts = []
    
    # System-level alerts
    if not status['is_running']:
        alerts.append("🔴 CRITICAL: Collector is not running")
    
    if status['health'] == 'critical':
        alerts.append("🚨 CRITICAL: System health is critical")
    elif status['health'] == 'warning':
        alerts.append("⚠️  WARNING: System health needs attention")
    
    # Check last update time
    if 'last_updated' in status:
        try:
            last_update = datetime.fromisoformat(status['last_updated'])
            now = datetime.now()
            
            # Handle timezone differences
            if last_update.tzinfo is not None and now.tzinfo is None:
                last_update = last_update.replace(tzinfo=None)
            
            minutes_since_update = (now - last_update).total_seconds() / 60
            
            if minutes_since_update > 120:  # 2 hours
                alerts.append(f"⚠️  WARNING: No status update for {minutes_since_update:.0f} minutes")
        except Exception:
            alerts.append("⚠️  WARNING: Cannot parse last update time")
    
    # Currency-specific alerts
    currencies = status.get('currencies', {})
    for currency, stats in currencies.items():
        
        # High failure rate
        if stats['success_rate'] < 50 and stats['total_runs'] > 5:
            alerts.append(f"🚨 {currency}: Low success rate ({stats['success_rate']:.1f}%)")
        
        # Consecutive failures
        if stats['consecutive_failures'] >= 5:
            alerts.append(f"🔴 {currency}: {stats['consecutive_failures']} consecutive failures")
        elif stats['consecutive_failures'] >= 3:
            alerts.append(f"⚠️  {currency}: {stats['consecutive_failures']} consecutive failures")
        
        # No recent success
        last_success = stats.get('last_success')
        if last_success:
            try:
                last_success_time = datetime.fromisoformat(last_success)
                now = datetime.now()
                
                if last_success_time.tzinfo is not None and now.tzinfo is None:
                    last_success_time = last_success_time.replace(tzinfo=None)
                
                hours_since_success = (now - last_success_time).total_seconds() / 3600
                
                if hours_since_success > 24:
                    alerts.append(f"🔴 {currency}: No success for {hours_since_success:.1f} hours")
                elif hours_since_success > 6:
                    alerts.append(f"⚠️  {currency}: No success for {hours_since_success:.1f} hours")
            except Exception:
                pass
        elif stats['total_runs'] > 3:
            alerts.append(f"🔴 {currency}: No successful collections yet")
    
    return alerts

def print_alerts_only(status: Dict[str, Any]) -> None:
    """Print only alerts and issues."""
    alerts = check_for_alerts(status)
    
    print("🚨 BITCOIN OPTIONS COLLECTOR - ALERTS")
    print("=" * 50)
    
    if not alerts:
        print("💚 NO ALERTS - System is healthy!")
        print(f"Status: {status['status'].upper()}")
        print(f"Health: {status['health'].upper()}")
        print(f"Running: {'Yes' if status['is_running'] else 'No'}")
    else:
        print(f"⚠️  {len(alerts)} ALERT(S) DETECTED:")
        print("-" * 30)
        for alert in alerts:
            print(alert)
    
    print("\nLast Update:", format_time_ago(status.get('last_updated', '')))

def watch_status(status_dir: str = "continuous_data", interval: int = 30) -> None:
    """
    Continuously monitor status with auto-refresh.
    
    Args:
        status_dir: Directory containing status files
        interval: Refresh interval in seconds
    """
    print("👀 CONTINUOUS MONITORING MODE")
    print(f"Refreshing every {interval} seconds. Press Ctrl+C to stop.")
    print("=" * 60)
    
    try:
        while True:
            # Clear screen (works on most terminals)
            print("\033[2J\033[H", end="")
            
            # Current time
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"🕐 Monitoring at {current_time}")
            print()
            
            # Load and display status
            status = load_collector_status(status_dir)
            
            if status:
                print_basic_status(status)
                
                # Show any alerts
                alerts = check_for_alerts(status)
                if alerts:
                    print("\n🚨 ACTIVE ALERTS:")
                    print("-" * 20)
                    for alert in alerts[:5]:  # Show max 5 alerts
                        print(alert)
                    if len(alerts) > 5:
                        print(f"... and {len(alerts) - 5} more alerts")
            else:
                print("❌ No status data available")
                print("   • Collector may not be running")
                print("   • Status files may not exist")
                print("   • Check continuous_data/status/ directory")
            
            print(f"\n⏰ Next refresh in {interval} seconds...")
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped by user")

def main():
    """Main monitor script entry point."""
    parser = argparse.ArgumentParser(
        description="Monitor Bitcoin Options Continuous Data Collector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python monitor_collector.py                    # Single status check
  python monitor_collector.py --watch            # Continuous monitoring
  python monitor_collector.py --detailed         # Detailed report
  python monitor_collector.py --alerts           # Show alerts only
  python monitor_collector.py --watch --interval 60  # Watch with 60s refresh
        """
    )
    
    parser.add_argument(
        '--watch', '-w',
        action='store_true',
        help='Continuously monitor with auto-refresh'
    )
    
    parser.add_argument(
        '--detailed', '-d',
        action='store_true', 
        help='Show detailed status information'
    )
    
    parser.add_argument(
        '--alerts', '-a',
        action='store_true',
        help='Show only alerts and issues'
    )
    
    parser.add_argument(
        '--interval', '-i',
        type=int,
        default=30,
        help='Refresh interval for watch mode (seconds, default: 30)'
    )
    
    parser.add_argument(
        '--status-dir',
        default='continuous_data',
        help='Directory containing status files (default: continuous_data)'
    )
    
    args = parser.parse_args()
    
    # Load status
    status = load_collector_status(args.status_dir)
    
    if not status:
        print("❌ No collector status found!")
        print()
        print("Possible causes:")
        print("  • Continuous collector is not running")
        print("  • Status directory doesn't exist")
        print("  • Status files are corrupted")
        print()
        print("Solutions:")
        print("  • Start collector: python src/continuous_collector.py")
        print("  • Test collector: python src/continuous_collector.py --test")
        print("  • Check directory: ls -la continuous_data/status/")
        return 1
    
    # Execute based on arguments
    try:
        if args.watch:
            watch_status(args.status_dir, args.interval)
        elif args.detailed:
            print_detailed_status(status)
        elif args.alerts:
            print_alerts_only(status)
        else:
            is_healthy = print_basic_status(status)
            return 0 if is_healthy else 1
            
    except KeyboardInterrupt:
        print("\n👋 Monitoring interrupted by user")
        return 0
    except Exception as e:
        print(f"\n❌ Monitor error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
