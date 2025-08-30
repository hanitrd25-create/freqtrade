#!/bin/bash
# Fix for the date column bug in freqtrade converter.py
# This fixes the issue where 'date' column is renamed to 'index' in ohlcv_fill_up_missing_data

echo "Applying fix for date column bug in converter.py..."

# Update converter.py to fix the date column issue
cat > /tmp/converter_fix.patch << 'EOF'
--- a/freqtrade/data/converter/converter.py
+++ b/freqtrade/data/converter/converter.py
@@ -117,6 +117,9 @@ def ohlcv_fill_up_missing_data(dataframe: DataFrame, timeframe: str, pair: str)
         }
     )
     df.reset_index(inplace=True)
+    # Fix: Ensure the date column is named 'date' not 'index'
+    if 'index' in df.columns and 'date' not in df.columns:
+        df.rename(columns={'index': 'date'}, inplace=True)
     len_before = len(dataframe)
     len_after = len(df)
     pct_missing = (len_after - len_before) / len_before if len_before > 0 else 0
EOF

# Apply the patch
cd /home/freqtrade
patch -p1 < /tmp/converter_fix.patch

# Update version
sed -i 's/__version__ = "2025.8-rc4"/__version__ = "2025.8-rc5"/' /home/freqtrade/freqtrade/__init__.py

# Clear Python cache
echo "Clearing Python cache..."
find /home/freqtrade -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find /home/freqtrade -name "*.pyc" -delete 2>/dev/null
find /home/freqtrade -name "*.pyo" -delete 2>/dev/null

echo "Fix applied successfully!"
echo ""
echo "The bug was in /home/freqtrade/freqtrade/data/converter/converter.py"
echo "The ohlcv_fill_up_missing_data function was renaming 'date' to 'index'"
echo ""
echo "Now test with:"
echo "  freqtrade backtesting --strategy DLIStrategy"
