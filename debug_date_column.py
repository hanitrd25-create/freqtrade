#!/usr/bin/env python3
"""
Debug script to trace where the 'date' column is being lost
Run this on your server to identify the exact point of failure
"""

import sys
import os

# Add freqtrade to path
sys.path.insert(0, '/home/freqtrade')
sys.path.insert(0, '/home/ubuntu')

def trace_date_column():
    """Trace the date column through the data loading pipeline"""
    
    from freqtrade.data.history import load_pair_history, load_data
    from freqtrade.data.history.datahandlers import get_datahandler
    from freqtrade.enums import CandleType
    from pathlib import Path
    
    # Test parameters
    pair = "BTC/USDT:USDT"
    timeframe = "5m"
    datadir = Path("/home/ubuntu/user_data/data/binance")
    candle_type = CandleType.FUTURES
    
    print("="*60)
    print("TRACING DATE COLUMN ISSUE")
    print("="*60)
    
    # Step 1: Test FeatherDataHandler directly
    print("\n1. Testing FeatherDataHandler directly:")
    data_handler = get_datahandler(datadir, "feather")
    
    # Monkey-patch to add debug output
    original_ohlcv_load = data_handler.ohlcv_load
    def debug_ohlcv_load(*args, **kwargs):
        result = original_ohlcv_load(*args, **kwargs)
        print(f"   After ohlcv_load: columns = {list(result.columns) if not result.empty else 'EMPTY'}")
        print(f"   Has 'date' column: {'date' in result.columns if not result.empty else 'N/A'}")
        return result
    data_handler.ohlcv_load = debug_ohlcv_load
    
    # Test direct loading
    df = data_handler.ohlcv_load(
        pair=pair,
        timeframe=timeframe,
        candle_type=candle_type
    )
    
    print(f"   Direct load result: {df.shape if not df.empty else 'EMPTY'}")
    print(f"   Columns: {list(df.columns) if not df.empty else 'EMPTY'}")
    
    # Step 2: Test load_pair_history
    print("\n2. Testing load_pair_history:")
    df2 = load_pair_history(
        datadir=datadir,
        timeframe=timeframe,
        pair=pair,
        data_handler=data_handler,
        candle_type=candle_type
    )
    print(f"   Result: {df2.shape if not df2.empty else 'EMPTY'}")
    print(f"   Columns: {list(df2.columns) if not df2.empty else 'EMPTY'}")
    print(f"   Has 'date' column: {'date' in df2.columns if not df2.empty else 'N/A'}")
    
    # Step 3: Test load_data (returns LazyDataLoader)
    print("\n3. Testing load_data (LazyDataLoader):")
    data = load_data(
        datadir=datadir,
        timeframe=timeframe,
        pairs=[pair],
        data_handler=data_handler,
        candle_type=candle_type
    )
    
    print(f"   LazyDataLoader created")
    print(f"   Accessing data for {pair}...")
    
    # Access the data
    try:
        pair_data = data[pair]
        print(f"   Result: {pair_data.shape}")
        print(f"   Columns: {list(pair_data.columns)}")
        print(f"   Has 'date' column: {'date' in pair_data.columns}")
        
        # Check the actual cached data
        if hasattr(data, '_cache'):
            print(f"\n   Cache contents:")
            for cached_pair, cached_df in data._cache.items():
                print(f"     {cached_pair}: columns = {list(cached_df.columns)}")
                print(f"     Has 'date': {'date' in cached_df.columns}")
        
    except KeyError as e:
        print(f"   ERROR: {e}")
    except Exception as e:
        print(f"   UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    # Step 4: Test get_timerange
    print("\n4. Testing get_timerange:")
    try:
        from freqtrade.data.history import get_timerange
        min_date, max_date = get_timerange(data)
        print(f"   SUCCESS: {min_date} to {max_date}")
    except KeyError as e:
        print(f"   ERROR: {e}")
        print(f"   This is where the failure happens!")
        
        # Debug the actual data
        print("\n   Debugging data structure:")
        for pair_name in [pair, "BTC/USDT:USDT", "BTC_USDT_USDT", "BTCUSDT"]:
            try:
                test_data = data[pair_name]
                print(f"   Found data for '{pair_name}': columns = {list(test_data.columns)}")
            except:
                pass
    except Exception as e:
        print(f"   UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    trace_date_column()
