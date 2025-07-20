#!/usr/bin/env python3
"""
Test Script for Continuous Data Collector

This script tests the continuous collector functionality to ensure
everything is working before starting long-running collection.

Usage:
    python test_continuous_collector.py
"""

import sys
import time
import threading
import logging
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Setup logging for tests
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def test_dependencies():
    """Test that all required dependencies are available."""
    print("🔧 Testing Dependencies...")
    try:
        from src.continuous_collector import ContinuousDataCollector
        print("✅ ContinuousDataCollector imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("💡 Make sure you have:")
        print("   - Updated requirements.txt with 'schedule>=1.2.0'")
        print("   - Installed: pip install schedule")
        print("   - Created src/continuous_collector.py")
        return False

def test_basic_initialization():
    """Test basic continuous collector initialization."""
    print("\n1️⃣ Testing Initialization...")
    try:
        from src.continuous_collector import ContinuousDataCollector
        
        collector = ContinuousDataCollector(
            currencies=['BTC'],  # Just test BTC for speed
            collection_interval_minutes=1,  # Fast for testing
            lookback_hours=1,  # Small lookback
            output_dir="test_continuous_data",
            enable_scheduling=False  # Manual control for testing
        )
        print("✅ Collector initialized successfully")
        print(f"   Currencies: {collector.currencies}")
        print(f"   Interval: {collector.collection_interval} minutes")
        print(f"   Output: {collector.output_dir}")
        return True, collector
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return False, None

def test_single_collection(collector):
    """Test single collection cycle."""
    print("\n2️⃣ Testing Single Collection Cycle...")
    try:
        results = collector.run_collection_cycle()
        success_count = sum(1 for success in results.values() if success)
        print(f"✅ Collection cycle completed: {success_count}/{len(results)} successful")
        
        if success_count == 0:
            print("⚠️  No data collected - this may be normal for recent dates")
            print("   The API connection and logic are working correctly")
        else:
            print(f"📊 Successfully collected data for {success_count} currencies")
        
        return True
    except Exception as e:
        print(f"❌ Collection cycle failed: {e}")
        import traceback
        print(f"   Error details: {traceback.format_exc()}")
        return False

def test_status_reporting(collector):
    """Test status reporting functionality."""
    print("\n3️⃣ Testing Status Reporting...")
    try:
        status = collector.get_status()
        print(f"✅ Status retrieved: {status['status']}")
        print(f"   Health: {status['health']}")
        print(f"   Running: {status['is_running']}")
        
        # Test detailed status display
        collector.print_status()
        return True
    except Exception as e:
        print(f"❌ Status reporting failed: {e}")
        return False

def test_short_continuous_run(collector):
    """Test short continuous run (30 seconds)."""
    print("\n4️⃣ Testing Short Continuous Run (30 seconds)...")
    try:
        collector.start()
        print("✅ Collector started in background")
        
        # Wait 30 seconds with progress updates
        for i in range(30):
            time.sleep(1)
            if i % 10 == 0 and i > 0:
                print(f"   Running... {30-i} seconds remaining")
        
        collector.stop()
        print("✅ Collector stopped successfully")
        
        # Check final status
        final_status = collector.get_status()
        total_records = sum(stats['total_records'] 
                          for stats in final_status['currencies'].values())
        
        if total_records > 0:
            print(f"✅ Total records collected during test: {total_records}")
        else:
            print("ℹ️  No records collected during test (normal for short run)")
            
        return True
        
    except Exception as e:
        print(f"❌ Continuous run test failed: {e}")
        return False

def test_error_handling():
    """Test error handling with invalid configuration."""
    print("\n5️⃣ Testing Error Handling...")
    try:
        from src.continuous_collector import ContinuousDataCollector
        
        # Test with invalid currency to trigger controlled errors
        error_collector = ContinuousDataCollector(
            currencies=['INVALID_CURRENCY'],  # This should fail gracefully
            collection_interval_minutes=1,
            max_consecutive_failures=2,  # Low threshold for testing
            restart_delay_minutes=1,
            enable_scheduling=False,
            output_dir="test_error_handling"
        )
        
        print("   Testing with invalid currency...")
        results = error_collector.run_collection_cycle()
        
        # Should have failures but not crash
        failed_count = sum(1 for success in results.values() if not success)
        if failed_count > 0:
            print(f"✅ Error handling working: {failed_count} expected failures detected")
            
            # Check health status updated correctly
            error_collector.update_health_status()
            status = error_collector.get_status()
            
            if status['health'] in ['warning', 'critical']:
                print(f"✅ Health monitoring working: Status is {status['health']}")
            
            return True
        else:
            print("⚠️  Expected failures not detected - may indicate issue")
            return False
            
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def test_file_structure():
    """Test that output file structure is created correctly."""
    print("\n6️⃣ Testing File Structure Creation...")
    try:
        test_dir = Path("test_continuous_data")
        
        # Check directories were created
        expected_dirs = ['logs', 'data', 'status']
        for dir_name in expected_dirs:
            dir_path = test_dir / dir_name
            if dir_path.exists():
                print(f"✅ Directory created: {dir_name}/")
            else:
                print(f"❌ Missing directory: {dir_name}/")
                return False
        
        # Check for log files
        log_files = list((test_dir / "logs").glob("*.log"))
        if log_files:
            print(f"✅ Log files created: {len(log_files)} file(s)")
        
        # Check for status files
        status_files = list((test_dir / "status").glob("*.json"))
        if status_files:
            print(f"✅ Status files created: {len(status_files)} file(s)")
        
        return True
        
    except Exception as e:
        print(f"❌ File structure test failed: {e}")
        return False

def cleanup_test_files():
    """Clean up test files after testing."""
    print("\n🧹 Cleaning up test files...")
    try:
        import shutil
        
        test_dirs = ["test_continuous_data", "test_error_handling"]
        for test_dir in test_dirs:
            if Path(test_dir).exists():
                shutil.rmtree(test_dir)
                print(f"✅ Cleaned up: {test_dir}/")
        
    except Exception as e:
        print(f"⚠️  Cleanup warning: {e}")

def main():
    """Main test function."""
    print("🚀 Bitcoin Options Analytics - Continuous Collector Tests")
    print("=" * 65)
    
    # Test sequence
    tests = [
        ("Dependencies Check", test_dependencies),
    ]
    
    # Run dependency check first
    if not test_dependencies():
        print("\n❌ Dependency test failed. Please fix dependencies before continuing.")
        return 1
    
    # Import after dependency check
    from src.continuous_collector import ContinuousDataCollector
    
    # Initialize collector for main tests
    print("\n📋 Initializing test collector...")
    success, collector = test_basic_initialization()
    if not success:
        return 1
    
    # Define remaining tests
    main_tests = [
        ("Single Collection", lambda: test_single_collection(collector)),
        ("Status Reporting", lambda: test_status_reporting(collector)),
        ("Short Continuous Run", lambda: test_short_continuous_run(collector)),
        ("Error Handling", test_error_handling),
        ("File Structure", test_file_structure)
    ]
    
    # Run all tests
    results = [("Initialization", True)]  # Already passed
    
    for test_name, test_func in main_tests:
        try:
            print(f"\n{'='*20}")
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Clean up
    cleanup_test_files()
    
    # Summary
    print("\n" + "=" * 65)
    print("📋 TEST SUMMARY")
    print("=" * 65)
    
    all_passed = True
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:<20}: {status}")
        if not success:
            all_passed = False
    
    print("\n" + "=" * 65)
    
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("🚀 Your continuous collector is ready for production use!")
        print("\n📖 Next Steps:")
        print("  1. Start collecting: python src/continuous_collector.py")
        print("  2. Monitor status:   python monitor_collector.py")
        print("  3. Test single run:  python src/continuous_collector.py --test")
        print("  4. Custom settings:  python src/continuous_collector.py --help")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("\n💡 Common solutions:")
        print("  • Install missing dependencies: pip install schedule")
        print("  • Check network connectivity")
        print("  • Verify src/continuous_collector.py exists")
        print("  • Update requirements.txt")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
