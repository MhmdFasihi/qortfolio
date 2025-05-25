#!/usr/bin/env python3
"""
Continuous Data Collector with Auto-Recovery
===========================================

This module provides a production-ready continuous data collection system
that automatically handles errors, recovers from failures, and runs without
manual intervention.

Features:
- Automatic error recovery and restart
- Health monitoring and alerting  
- Scheduled data collection
- Status reporting and logging
- Rate limiting and connection management
- Graceful shutdown handling

Usage:
    python src/continuous_collector.py --currencies BTC ETH --interval 30
"""

import asyncio
import logging
import signal
import sys
import time
import threading
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import schedule
from dataclasses import dataclass, field
from enum import Enum
import traceback

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.data.collectors import DeribitCollector, DataCollectionError
    from src.utils.time_utils import datetime_to_timestamp
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)

class CollectorStatus(Enum):
    """Status states for the collector."""
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    STOPPING = "stopping"
    STOPPED = "stopped"

class HealthStatus(Enum):
    """Health status indicators."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

@dataclass
class CollectionStats:
    """Statistics for data collection performance."""
    total_runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    total_records: int = 0
    last_run_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    consecutive_failures: int = 0
    uptime_start: datetime = field(default_factory=datetime.now)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_runs == 0:
            return 0.0
        return (self.successful_runs / self.total_runs) * 100
    
    @property
    def uptime_hours(self) -> float:
        """Calculate uptime in hours."""
        return (datetime.now() - self.uptime_start).total_seconds() / 3600

class ContinuousDataCollector:
    """
    Production-ready continuous data collector with auto-recovery.
    
    This collector runs indefinitely, automatically recovering from errors
    and providing comprehensive monitoring and logging.
    """
    
    def __init__(self, 
                 currencies: List[str] = None,
                 collection_interval_minutes: int = 30,
                 lookback_hours: int = 2,
                 output_dir: str = "continuous_data",
                 max_consecutive_failures: int = 10,
                 restart_delay_minutes: int = 5,
                 enable_scheduling: bool = True):
        """
        Initialize continuous data collector.
        
        Args:
            currencies: List of currencies to collect (default: ['BTC', 'ETH'])
            collection_interval_minutes: Minutes between collections
            lookback_hours: Hours of historical data to collect each run
            output_dir: Directory to save collected data
            max_consecutive_failures: Max failures before extended pause
            restart_delay_minutes: Minutes to wait before restart after failure
            enable_scheduling: Whether to use scheduler (vs manual control)
        """
        self.currencies = currencies or ['BTC', 'ETH']
        self.collection_interval = collection_interval_minutes
        self.lookback_hours = lookback_hours
        self.output_dir = Path(output_dir)
        self.max_consecutive_failures = max_consecutive_failures
        self.restart_delay = restart_delay_minutes
        self.enable_scheduling = enable_scheduling
        
        # Create output directories
        self.output_dir.mkdir(exist_ok=True, parents=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)
        (self.output_dir / "status").mkdir(exist_ok=True)
        
        # Status tracking
        self.status = CollectorStatus.STOPPED
        self.health = HealthStatus.UNKNOWN
        self.stats = {currency: CollectionStats() for currency in self.currencies}
        self.is_running = False
        self.should_stop = False
        
        # Threading
        self.main_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        
        # Setup logging
        self.setup_logging()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        self.logger.info(f"ContinuousDataCollector initialized for {self.currencies}")
        self.logger.info(f"Collection interval: {self.collection_interval} minutes")
        self.logger.info(f"Output directory: {self.output_dir}")

    def setup_logging(self):
        """Setup comprehensive logging system."""
        log_file = self.output_dir / "logs" / f"collector_{datetime.now().strftime('%Y%m%d')}.log"
        
        # Create logger
        self.logger = logging.getLogger('ContinuousCollector')
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # File handler with rotation
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        signal_names = {signal.SIGINT: "SIGINT", signal.SIGTERM: "SIGTERM"}
        signal_name = signal_names.get(signum, f"Signal {signum}")
        
        self.logger.info(f"Received {signal_name}, initiating graceful shutdown...")
        self.stop()

    def collect_data_for_currency(self, currency: str) -> bool:
        """
        Collect data for a specific currency with comprehensive error handling.
        
        Args:
            currency: Currency to collect data for
            
        Returns:
            True if collection successful, False otherwise
        """
        stats = self.stats[currency]
        stats.total_runs += 1
        stats.last_run_time = datetime.now()
        
        try:
            self.logger.info(f"Starting data collection for {currency}")
            
            # Calculate time range
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=self.lookback_hours)
            
            # Create collector with enhanced error handling
            with DeribitCollector() as collector:
                # Test connection first
                if not collector.test_connection():
                    raise DataCollectionError("API connection test failed")
                
                # Collect data
                data = collector.collect_options_data(
                    currency=currency,
                    start_date=start_time.date(),
                    end_date=end_time.date()
                )
                
                if data.empty:
                    self.logger.warning(f"No data collected for {currency}")
                    # Not considered a failure if API is working
                    stats.successful_runs += 1
                    stats.consecutive_failures = 0
                    return True
                
                # Save data with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{currency}_options_{timestamp}.csv"
                filepath = self.output_dir / "data" / filename
                
                data.to_csv(filepath, index=False)
                
                # Update statistics
                stats.successful_runs += 1
                stats.total_records += len(data)
                stats.last_success_time = datetime.now()
                stats.consecutive_failures = 0
                
                self.logger.info(f"✅ {currency}: Collected {len(data)} records, saved to {filename}")
                return True
                
        except Exception as e:
            stats.failed_runs += 1
            stats.consecutive_failures += 1
            
            error_msg = f"❌ {currency}: Collection failed - {str(e)}"
            self.logger.error(error_msg)
            
            # Log full traceback for debugging
            self.logger.debug(f"Full traceback for {currency}:\n{traceback.format_exc()}")
            
            return False

    def run_collection_cycle(self):
        """Run a complete data collection cycle for all currencies."""
        cycle_start = datetime.now()
        self.logger.info(f"🔄 Starting collection cycle at {cycle_start}")
        
        results = {}
        total_success = 0
        
        for currency in self.currencies:
            if self.should_stop:
                break
                
            try:
                success = self.collect_data_for_currency(currency)
                results[currency] = success
                if success:
                    total_success += 1
                    
            except Exception as e:
                self.logger.error(f"Unexpected error collecting {currency}: {e}")
                results[currency] = False
        
        # Update overall health status
        self.update_health_status()
        
        # Log cycle summary
        cycle_duration = (datetime.now() - cycle_start).total_seconds()
        self.logger.info(f"📊 Cycle completed: {total_success}/{len(self.currencies)} currencies successful")
        self.logger.info(f"⏱️  Cycle duration: {cycle_duration:.1f} seconds")
        
        # Save status
        self.save_status()
        
        return results

    def update_health_status(self):
        """Update overall health status based on recent performance."""
        with self.lock:
            total_failures = sum(stats.consecutive_failures for stats in self.stats.values())
            max_failures = max(stats.consecutive_failures for stats in self.stats.values())
            
            if max_failures == 0:
                self.health = HealthStatus.HEALTHY
            elif max_failures < 3:
                self.health = HealthStatus.WARNING
            elif max_failures >= self.max_consecutive_failures:
                self.health = HealthStatus.CRITICAL
            else:
                self.health = HealthStatus.WARNING

    def should_pause_collection(self) -> bool:
        """Determine if collection should be paused due to excessive failures."""
        max_failures = max(stats.consecutive_failures for stats in self.stats.values())
        return max_failures >= self.max_consecutive_failures

    def recovery_pause(self):
        """Pause collection for recovery after excessive failures."""
        pause_minutes = self.restart_delay * (1 + max(stats.consecutive_failures 
                                                    for stats in self.stats.values()) // 5)
        pause_minutes = min(pause_minutes, 60)  # Cap at 1 hour
        
        self.logger.warning(f"⏸️  Entering recovery pause for {pause_minutes} minutes")
        self.status = CollectorStatus.PAUSED
        
        for i in range(pause_minutes * 60):  # Check every second for stop signal
            if self.should_stop:
                break
            time.sleep(1)
        
        self.logger.info("▶️  Recovery pause completed, resuming collection")

    def run_continuous(self):
        """Main continuous collection loop."""
        self.logger.info("🚀 Starting continuous data collection")
        self.status = CollectorStatus.STARTING
        
        try:
            self.is_running = True
            self.status = CollectorStatus.RUNNING
            
            if self.enable_scheduling:
                # Setup scheduler
                schedule.every(self.collection_interval).minutes.do(self.run_collection_cycle)
                
                # Run initial collection
                self.run_collection_cycle()
                
                # Main scheduler loop
                while not self.should_stop:
                    try:
                        # Check if we should pause due to failures
                        if self.should_pause_collection():
                            self.recovery_pause()
                            continue
                        
                        # Run scheduled jobs
                        schedule.run_pending()
                        time.sleep(1)
                        
                    except Exception as e:
                        self.logger.error(f"Scheduler error: {e}")
                        self.status = CollectorStatus.ERROR
                        time.sleep(10)  # Brief pause before continuing
                        
            else:
                # Manual control mode - run immediately and continuously
                while not self.should_stop:
                    try:
                        if self.should_pause_collection():
                            self.recovery_pause()
                            continue
                        
                        self.run_collection_cycle()
                        
                        # Wait for next cycle
                        for _ in range(self.collection_interval * 60):
                            if self.should_stop:
                                break
                            time.sleep(1)
                            
                    except Exception as e:
                        self.logger.error(f"Collection loop error: {e}")
                        self.status = CollectorStatus.ERROR
                        time.sleep(30)  # Recovery pause
                        
        except Exception as e:
            self.logger.error(f"Fatal error in continuous collection: {e}")
            self.status = CollectorStatus.ERROR
            raise
        
        finally:
            self.is_running = False
            self.status = CollectorStatus.STOPPED
            self.logger.info("🛑 Continuous data collection stopped")

    def start(self):
        """Start the continuous collector in a separate thread."""
        if self.is_running:
            self.logger.warning("Collector is already running")
            return
        
        self.should_stop = False
        self.main_thread = threading.Thread(target=self.run_continuous, daemon=False)
        self.main_thread.start()
        
        self.logger.info("Continuous collector started in background thread")

    def stop(self):
        """Stop the continuous collector gracefully."""
        self.logger.info("Stopping continuous collector...")
        self.should_stop = True
        self.status = CollectorStatus.STOPPING
        
        if self.main_thread and self.main_thread.is_alive():
            self.main_thread.join(timeout=30)
            if self.main_thread.is_alive():
                self.logger.warning("Collector thread did not stop gracefully")
        
        self.save_status()
        self.logger.info("Continuous collector stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get current collector status and statistics."""
        with self.lock:
            return {
                "status": self.status.value,
                "health": self.health.value,
                "uptime_hours": max(stats.uptime_hours for stats in self.stats.values()),
                "currencies": {
                    currency: {
                        "total_runs": stats.total_runs,
                        "success_rate": stats.success_rate,
                        "total_records": stats.total_records,
                        "consecutive_failures": stats.consecutive_failures,
                        "last_success": stats.last_success_time.isoformat() if stats.last_success_time else None
                    }
                    for currency, stats in self.stats.items()
                },
                "is_running": self.is_running,
                "should_stop": self.should_stop
            }

    def save_status(self):
        """Save current status to file."""
        try:
            status_file = self.output_dir / "status" / "collector_status.json"
            status_data = self.get_status()
            status_data["last_updated"] = datetime.now().isoformat()
            
            with open(status_file, 'w') as f:
                json.dump(status_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save status: {e}")

    def print_status(self):
        """Print current status to console."""
        status = self.get_status()
        
        print("\n" + "="*60)
        print("🚀 CONTINUOUS DATA COLLECTOR STATUS")
        print("="*60)
        print(f"Status: {status['status'].upper()}")
        print(f"Health: {status['health'].upper()}")
        print(f"Uptime: {status['uptime_hours']:.1f} hours")
        print(f"Running: {'✅ YES' if status['is_running'] else '❌ NO'}")
        print()
        
        print("📊 CURRENCY STATISTICS:")
        print("-" * 40)
        for currency, stats in status['currencies'].items():
            print(f"{currency}:")
            print(f"  • Total runs: {stats['total_runs']}")
            print(f"  • Success rate: {stats['success_rate']:.1f}%")
            print(f"  • Records collected: {stats['total_records']:,}")
            print(f"  • Consecutive failures: {stats['consecutive_failures']}")
            print(f"  • Last success: {stats['last_success'] or 'Never'}")
            print()
        
        print("="*60)


def main():
    """Main entry point for continuous collector."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Continuous Bitcoin Options Data Collector")
    parser.add_argument('--currencies', nargs='+', default=['BTC', 'ETH'],
                       help='Currencies to collect (default: BTC ETH)')
    parser.add_argument('--interval', type=int, default=30,
                       help='Collection interval in minutes (default: 30)')
    parser.add_argument('--lookback', type=int, default=2,
                       help='Hours of historical data to collect (default: 2)')
    parser.add_argument('--output-dir', default='continuous_data',
                       help='Output directory (default: continuous_data)')
    parser.add_argument('--max-failures', type=int, default=10,
                       help='Max consecutive failures before pause (default: 10)')
    parser.add_argument('--restart-delay', type=int, default=5,
                       help='Minutes to wait before restart (default: 5)')
    parser.add_argument('--no-schedule', action='store_true',
                       help='Disable scheduler, run continuously')
    parser.add_argument('--test', action='store_true',
                       help='Run a single test collection and exit')
    
    args = parser.parse_args()
    
    # Create collector
    collector = ContinuousDataCollector(
        currencies=args.currencies,
        collection_interval_minutes=args.interval,
        lookback_hours=args.lookback,
        output_dir=args.output_dir,
        max_consecutive_failures=args.max_failures,
        restart_delay_minutes=args.restart_delay,
        enable_scheduling=not args.no_schedule
    )
    
    if args.test:
        # Run single test collection
        print("🧪 Running test collection...")
        results = collector.run_collection_cycle()
        collector.print_status()
        
        success_count = sum(1 for success in results.values() if success)
        print(f"\n🎯 Test Results: {success_count}/{len(results)} currencies successful")
        return 0 if success_count > 0 else 1
    
    try:
        # Start continuous collection
        collector.start()
        
        print("🚀 Continuous collector started!")
        print("Press Ctrl+C to stop gracefully")
        print("Use --test flag to run a single test collection")
        
        # Keep main thread alive and provide status updates
        while collector.is_running:
            time.sleep(60)  # Status update every minute
            if collector.is_running:  # Check again after sleep
                collector.print_status()
        
    except KeyboardInterrupt:
        print("\n⚠️  Shutdown requested by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        return 1
    finally:
        collector.stop()
    
    print("✅ Continuous collector stopped successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())
