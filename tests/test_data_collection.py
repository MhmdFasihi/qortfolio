#!/usr/bin/env python3
"""
Simple test script for data collection functionality.
Run this to test if everything is working properly.
"""

import logging
import sys
from datetime import date, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def test_time_utilities():
    """Test time calculation utilities."""
    print("🕐 Testing Time Utilities...")
    try:
        from src.utils.time_utils import test_time_calculations
        success = test_time_calculations()
        print(f"   Result: {'✅ PASSED' if success else '❌ FAILED'}")
        return success
    except Exception as e:
        print(f"   Result: ❌ FAILED - {e}")
        return False

def test_api_connection():
    """Test API connection."""
    print("🌐 Testing API Connection...")
    try:
        from src.data.collectors import DeribitCollector
        
        collector = DeribitCollector()
        success = collector.test_connection()
        print(f"   Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
        return success
    except Exception as e:
        print(f"   Result: ❌ FAILED - {e}")
        return False

def test_data_collection():
    """Test actual data collection."""
    print("📊 Testing Data Collection...")
    try:
        from src.data.collectors import collect_btc_options
        
        # Use dates we know have data (historical dates)
        end_date = date(2024, 12, 31)   # End of 2024
        start_date = date(2024, 12, 30)  # 2 days for better chance of data
        
        print(f"   Collecting data from {start_date} to {end_date} (historical data)...")
        data = collect_btc_options(start_date=start_date, end_date=end_date)
        
        if not data.empty:
            print(f"   ✅ SUCCESS: Collected {len(data)} option records")
            print(f"   📈 Columns: {list(data.columns)}")
            
            if len(data) > 0:
                sample = data.iloc[0]
                print(f"   📊 Sample record:")
                print(f"      Instrument: {sample['instrument_name']}")
                print(f"      Price: ${sample['price']:.2f}")
                print(f"      Strike: ${sample['strike_price']:.0f}")
                print(f"      TTM: {sample['time_to_maturity']:.4f} years")
                print(f"      IV: {sample['implied_volatility']:.1%}")
            
            return True
        else:
            print("   ⚠️  No data collected - trying alternative date range...")
            
            # Try a different date range if first one fails
            end_date = date(2024, 11, 30)
            start_date = date(2024, 11, 29)
            print(f"   Trying {start_date} to {end_date}...")
            
            data = collect_btc_options(start_date=start_date, end_date=end_date)
            if not data.empty:
                print(f"   ✅ SUCCESS with alternative dates: Collected {len(data)} option records")
                return True
            else:
                print("   ⚠️  No data with alternative dates either - API might be working but no historical data available")
                return False
            
    except Exception as e:
        print(f"   ❌ FAILED - {e}")
        print("   (This might be due to API rate limiting or data availability)")
        return False

def main():
    """Run all tests."""
    print("🚀 Bitcoin Options Analytics - Data Collection Tests")
    print("=" * 60)
    
    tests = [
        ("Time Utilities", test_time_utilities),
        ("API Connection", test_api_connection),
        ("Data Collection", test_data_collection)
    ]
    
    results = []
    for test_name, test_func in tests:
        success = test_func()
        results.append((test_name, success))
        print()
    
    # Summary
    print("📋 Test Summary:")
    print("-" * 30)
    
    all_passed = True
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if not success:
            all_passed = False
    
    print("-" * 30)
    if all_passed:
        print("🎉 All tests passed! Ready to build Black-Scholes models.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())