#!/usr/bin/env python3
"""
Direct fix for FeatherDataHandler - apply this on the server
This replaces the _ohlcv_load method with a robust version
"""

import sys
import os

# The fixed _ohlcv_load method code
FIXED_CODE = '''    def _ohlcv_load(
        self, pair: str, timeframe: str, timerange: TimeRange | None, candle_type: CandleType
    ) -> DataFrame:
        """
        Internal method used to load data for one pair from disk.
        Implements the loading and conversion to DataFrame.
        Timerange trimming and dataframe validation happens outside of this method.
        :param pair: Pair to load data
        :param timeframe: Timeframe (e.g. "5m")
        :param timerange: Limit data to be loaded to this timerange.
                        Optionally implemented by subclasses to avoid loading
                        all data where possible.
        :param candle_type: Any of the enum CandleType (must match trading mode!)
        :return: DataFrame with ohlcv data, or empty DataFrame
        """
        if candle_type not in self._pairs_cache:
            self._pairs_cache[candle_type] = {}
        
        pair_key = (pair, timeframe, candle_type)
        if pair_key in self._pairs_cache[candle_type]:
            return self._pairs_cache[candle_type][pair_key].copy()
        
        if timerange:
            filename = self._pair_data_filename(
                self._datadir, pair, timeframe, candle_type, timerange=timerange
            )
        else:
            filename = self._pair_data_filename(
                self._datadir, pair, timeframe, candle_type=candle_type, no_timeframe_modify=True
            )
            if not filename.exists():
                return DataFrame(columns=self._columns)
        
        try:
            # Try multiple methods to read the feather file
            import pyarrow.feather as feather
            import pandas as pd
            
            # Method 1: Try reading with PyArrow
            try:
                table = feather.read_table(str(filename))
                pairdata = table.to_pandas()
            except:
                # Method 2: Fallback to pandas feather reader
                pairdata = pd.read_feather(filename)
            
            # CRITICAL: Handle column naming
            original_columns = list(pairdata.columns)
            print(f"[FEATHER FIX] Loaded {filename.name} with columns: {original_columns}")
            
            # Check if columns match expected format
            expected_columns = ["date", "open", "high", "low", "close", "volume"]
            
            # If columns are numeric (0, 1, 2, 3, 4, 5) or don't match, rename them
            if original_columns != expected_columns:
                if len(original_columns) == 6:
                    print(f"[FEATHER FIX] Renaming columns from {original_columns} to {expected_columns}")
                    pairdata.columns = expected_columns
                else:
                    print(f"[FEATHER FIX] ERROR: Wrong number of columns: {len(original_columns)}")
                    return DataFrame(columns=expected_columns)
            
            # Ensure date column exists
            if "date" not in pairdata.columns:
                print(f"[FEATHER FIX] CRITICAL: No date column found!")
                if len(pairdata.columns) == 6:
                    print(f"[FEATHER FIX] Force-setting column names")
                    pairdata.columns = expected_columns
            
            # Convert date column to datetime if needed
            if "date" in pairdata.columns:
                if not pd.api.types.is_datetime64_any_dtype(pairdata["date"]):
                    pairdata["date"] = pd.to_datetime(pairdata["date"], unit="ms", utc=True)
            
            # Cache the result
            self._pairs_cache[candle_type][pair_key] = pairdata.copy()
            
            return pairdata
            
        except Exception as e:
            print(f"[FEATHER FIX] ERROR loading {filename}: {e}")
            import traceback
            traceback.print_exc()
            return DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
'''

def main():
    # Find the featherdatahandler.py file
    feather_file = "/home/freqtrade/freqtrade/data/history/datahandlers/featherdatahandler.py"
    
    if not os.path.exists(feather_file):
        print(f"ERROR: File not found: {feather_file}")
        sys.exit(1)
    
    # Read the current file
    with open(feather_file, 'r') as f:
        content = f.read()
    
    # Find the _ohlcv_load method
    import_line = "from pandas import DataFrame, to_datetime"
    if import_line not in content:
        # Add pandas import if missing
        content = content.replace("from pandas import DataFrame", 
                                  "from pandas import DataFrame, to_datetime")
    
    # Find and replace the _ohlcv_load method
    start_marker = "    def _ohlcv_load("
    end_marker = "            return DataFrame(columns=self._columns)"
    
    start_idx = content.find(start_marker)
    if start_idx == -1:
        print("ERROR: Could not find _ohlcv_load method")
        sys.exit(1)
    
    # Find the end of the method (next method or end of class)
    next_method_idx = content.find("\n    def ", start_idx + len(start_marker))
    if next_method_idx == -1:
        next_method_idx = len(content)
    
    # Replace the method
    new_content = content[:start_idx] + FIXED_CODE + content[next_method_idx:]
    
    # Backup the original file
    import shutil
    backup_file = feather_file + ".backup"
    shutil.copy(feather_file, backup_file)
    print(f"Backed up original file to: {backup_file}")
    
    # Write the fixed version
    with open(feather_file, 'w') as f:
        f.write(new_content)
    
    print(f"Fixed {feather_file}")
    
    # Clear Python cache
    import subprocess
    subprocess.run("find /home/freqtrade -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null", 
                   shell=True)
    subprocess.run("find /home/freqtrade -name '*.pyc' -delete 2>/dev/null", 
                   shell=True)
    print("Cleared Python cache")
    
    # Update version
    version_file = "/home/freqtrade/freqtrade/__init__.py"
    with open(version_file, 'r') as f:
        version_content = f.read()
    
    version_content = version_content.replace('__version__ = "2025.8-rc3"', 
                                               '__version__ = "2025.8-rc4-fixed"')
    
    with open(version_file, 'w') as f:
        f.write(version_content)
    
    print("Updated version to 2025.8-rc4-fixed")
    print("\nFix applied successfully!")
    print("Now run: freqtrade backtesting --strategy DLIStrategy")

if __name__ == "__main__":
    main()
