#!/usr/bin/env python3
"""
Diagnostic script to inspect feather files and identify column issues.
Run this on your server to understand the structure of your feather files.
"""

import sys
import os
from pathlib import Path
import pyarrow.feather as feather
import pandas as pd

def inspect_feather_file(filepath):
    """Inspect a single feather file and report its structure."""
    print(f"\n{'='*60}")
    print(f"Inspecting: {filepath}")
    print(f"{'='*60}")
    
    try:
        # Read with PyArrow directly
        table = feather.read_table(str(filepath))
        print(f"PyArrow column names: {table.column_names}")
        print(f"PyArrow schema: {table.schema}")
        
        # Convert to pandas with default settings
        df_default = table.to_pandas()
        print(f"\nPandas (default) columns: {list(df_default.columns)}")
        print(f"Pandas (default) dtypes:\n{df_default.dtypes}")
        
        # Convert to pandas with Arrow dtypes
        df_arrow = table.to_pandas(types_mapper=lambda t: pd.ArrowDtype(t))
        print(f"\nPandas (Arrow dtypes) columns: {list(df_arrow.columns)}")
        print(f"Pandas (Arrow dtypes) dtypes:\n{df_arrow.dtypes}")
        
        # Check if columns are numeric
        print(f"\nColumn types:")
        for col in df_default.columns:
            print(f"  {repr(col)}: type={type(col).__name__}, is_int={isinstance(col, int)}")
        
        # Show first few rows
        print(f"\nFirst 3 rows (default):")
        print(df_default.head(3))
        
        # Check for 'date' column
        if 'date' in df_default.columns:
            print(f"\n✅ 'date' column found")
        else:
            print(f"\n❌ 'date' column NOT found")
            print(f"Available columns: {list(df_default.columns)}")
        
        return df_default.columns.tolist()
        
    except Exception as e:
        print(f"ERROR reading file: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    # Determine data directory
    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1])
    else:
        # Try common locations
        possible_dirs = [
            Path("/home/ubuntu/user_data/data/binance"),
            Path("user_data/data/binance"),
            Path("~/user_data/data/binance").expanduser(),
        ]
        
        data_dir = None
        for pdir in possible_dirs:
            if pdir.exists():
                data_dir = pdir
                break
        
        if not data_dir:
            print("Could not find data directory. Please provide as argument:")
            print("  python diagnose_feather_columns.py /path/to/data/dir")
            sys.exit(1)
    
    print(f"Scanning directory: {data_dir}")
    
    # Find all feather files
    feather_files = list(data_dir.glob("**/*.feather"))
    
    if not feather_files:
        print(f"No .feather files found in {data_dir}")
        sys.exit(1)
    
    print(f"Found {len(feather_files)} feather files")
    
    # Inspect a sample of files
    column_structures = {}
    for i, filepath in enumerate(feather_files[:5]):  # Inspect first 5 files
        cols = inspect_feather_file(filepath)
        if cols:
            cols_key = str(cols)
            if cols_key not in column_structures:
                column_structures[cols_key] = []
            column_structures[cols_key].append(filepath.name)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Unique column structures found: {len(column_structures)}")
    for cols_str, files in column_structures.items():
        print(f"\nColumn structure: {cols_str}")
        print(f"Files with this structure: {files[:3]}...")  # Show first 3 files
    
    # Test the IPC utility if available
    print(f"\n{'='*60}")
    print("Testing IPC Utility")
    print(f"{'='*60}")
    
    try:
        # Add freqtrade to path
        sys.path.insert(0, '/home/freqtrade')
        from freqtrade.data.ipc_utils import read_compressed_ipc_to_pandas
        
        test_file = feather_files[0]
        print(f"Testing read_compressed_ipc_to_pandas on: {test_file}")
        df = read_compressed_ipc_to_pandas(test_file)
        print(f"Result columns: {list(df.columns)}")
        print(f"Has 'date' column: {'date' in df.columns}")
        
    except ImportError as e:
        print(f"Could not import IPC utility: {e}")
    except Exception as e:
        print(f"Error testing IPC utility: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
