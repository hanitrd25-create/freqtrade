#!/bin/bash
# Direct patch script to fix FeatherDataHandler on server

echo "Applying FeatherDataHandler fix directly on server..."

# First, clear Python cache
echo "Clearing Python cache..."
find /home/freqtrade -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find /home/freqtrade -name "*.pyc" -delete 2>/dev/null
find /home/freqtrade -name "*.pyo" -delete 2>/dev/null

# Create the patch file
cat > /tmp/feather_fix.patch << 'EOF'
--- a/freqtrade/data/history/datahandlers/featherdatahandler.py
+++ b/freqtrade/data/history/datahandlers/featherdatahandler.py
@@ -52,13 +52,54 @@ class FeatherDataHandler(IDataHandler):
         else:
             filename = self._pair_data_filename(
                 self._datadir, pair, timeframe, candle_type=candle_type, no_timeframe_modify=True
             )
             if not filename.exists():
                 return DataFrame(columns=self._columns)
         try:
-            pairdata = feather.read_feather(filename)
-            pairdata.columns = self._columns
+            # Use optimized compressed IPC reading method from centralized utility
+            from freqtrade.data.ipc_utils import read_compressed_ipc_to_pandas
+            pairdata = read_compressed_ipc_to_pandas(filename)
+            
+            # Log what we received for debugging
+            original_columns = list(pairdata.columns)
+            logger.info(f"FeatherDataHandler: Loaded {filename.name} with columns: {original_columns}")
+            
+            # Handle column naming robustly
+            # Case 1: Columns are already correct (date, open, high, low, close, volume)
+            if original_columns == list(self._columns):
+                # Columns are already perfect, don't touch them
+                logger.debug(f"Columns already match expected format")
+                pass
+            # Case 2: Columns don't match but we have the right count
+            elif len(original_columns) == len(self._columns):
+                # Always assign our expected column names if they don't match
+                logger.info(f"Renaming columns from {original_columns} to {list(self._columns)}")
+                pairdata.columns = self._columns
+            else:
+                # Wrong number of columns - this is an error
+                logger.error(f"File {filename} has {len(original_columns)} columns, expected {len(self._columns)}")
+                logger.error(f"Columns found: {original_columns}")
+                return DataFrame(columns=self._columns)
+            
+            # CRITICAL FIX: Ensure we ALWAYS have a date column
+            if "date" not in pairdata.columns:
+                logger.error(f"CRITICAL: No 'date' column after processing {filename}")
+                logger.error(f"Original columns: {original_columns}")
+                logger.error(f"Current columns: {list(pairdata.columns)}")
+                logger.error(f"Expected columns: {list(self._columns)}")
+                # FORCE column names to recover - this is critical for backtesting
+                if len(pairdata.columns) == len(self._columns):
+                    logger.warning(f"FORCING column names to expected format")
+                    pairdata.columns = self._columns
+                elif len(pairdata.columns) == 6 and all(isinstance(col, int) for col in pairdata.columns):
+                    # Handle numeric column indices (0, 1, 2, 3, 4, 5)
+                    logger.warning(f"Detected numeric columns, forcing to expected format")
+                    pairdata.columns = self._columns
+                else:
+                    logger.error(f"Cannot recover - returning empty dataframe")
+                    return DataFrame(columns=self._columns)
+            
+            # Convert date column if needed
+            if "date" in pairdata.columns:
+                # Check if date is already in datetime format
+                import pandas as pd
+                if not pd.api.types.is_datetime64_any_dtype(pairdata["date"]):
+                    pairdata["date"] = to_datetime(pairdata["date"], unit="ms", utc=True)
+            
         except Exception as e:
             logger.error(
                 f"Could not load data for {pair} with timeframe {timeframe}: {e}.\n"
                 f"Reinitializing pair data folder for {pair}."
             )
             # Remove this file since its contents are faulty
EOF

# Apply the patch
echo "Applying patch to featherdatahandler.py..."
cd /home/freqtrade
patch -p1 < /tmp/feather_fix.patch

# Update version
echo "Updating version to 2025.8-rc4..."
sed -i 's/__version__ = "2025.8-rc3"/__version__ = "2025.8-rc4"/' /home/freqtrade/freqtrade/__init__.py

# Verify the changes
echo "Verifying changes..."
grep -n "FeatherDataHandler: Loaded" /home/freqtrade/freqtrade/data/history/datahandlers/featherdatahandler.py
grep "__version__" /home/freqtrade/freqtrade/__init__.py

echo "Fix applied! Now test with:"
echo "freqtrade backtesting --strategy DLIStrategy"
