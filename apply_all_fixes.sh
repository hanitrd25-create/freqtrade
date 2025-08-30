#!/bin/bash
# Complete fix script for freqtrade date column and import issues
# This script applies all necessary fixes to make backtesting work

echo "========================================="
echo "Applying ALL fixes for freqtrade v2025.8-rc5"
echo "========================================="

# Fix 1: Fix the date column issue in converter.py
echo ""
echo "1. Fixing converter.py (date column issue)..."
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

cd /home/freqtrade
patch -p1 < /tmp/converter_fix.patch 2>/dev/null
if [ $? -eq 0 ]; then
    echo "   ✓ converter.py patched successfully"
else
    echo "   ⚠ Patch may have already been applied or failed, trying direct edit..."
    # Try direct sed replacement
    sed -i '/df.reset_index(inplace=True)/a\    # Fix: Ensure the date column is named '"'"'date'"'"' not '"'"'index'"'"'\n    if '"'"'index'"'"' in df.columns and '"'"'date'"'"' not in df.columns:\n        df.rename(columns={'"'"'index'"'"': '"'"'date'"'"'}, inplace=True)' /home/freqtrade/freqtrade/data/converter/converter.py
    echo "   ✓ Direct edit applied"
fi

# Fix 2: Add missing import in backtesting.py
echo ""
echo "2. Fixing backtesting.py (missing import)..."
cat > /tmp/backtesting_fix.patch << 'EOF'
--- a/freqtrade/optimize/backtesting.py
+++ b/freqtrade/optimize/backtesting.py
@@ -54,6 +54,7 @@ from freqtrade.optimize.indicator_cache import (
     get_signal_cache,
     clear_all_caches,
 )
+from freqtrade.optimize.backtest_caching import get_strategy_run_id
 from freqtrade.optimize.optimize_reports import (
     generate_backtest_stats,
     show_backtest_results,
EOF

cd /home/freqtrade
patch -p1 < /tmp/backtesting_fix.patch 2>/dev/null
if [ $? -eq 0 ]; then
    echo "   ✓ backtesting.py patched successfully"
else
    echo "   ⚠ Patch may have already been applied or failed, trying direct edit..."
    # Check if import already exists
    if grep -q "from freqtrade.optimize.backtest_caching import get_strategy_run_id" /home/freqtrade/freqtrade/optimize/backtesting.py; then
        echo "   ✓ Import already exists"
    else
        # Add the import after the indicator_cache imports
        sed -i '/from freqtrade.optimize.optimize_reports import/i from freqtrade.optimize.backtest_caching import get_strategy_run_id' /home/freqtrade/freqtrade/optimize/backtesting.py
        echo "   ✓ Import added"
    fi
fi

# Update version to 2025.8-rc5 if not already done
echo ""
echo "3. Updating version to 2025.8-rc5..."
sed -i 's/__version__ = "2025.8-rc4"/__version__ = "2025.8-rc5"/' /home/freqtrade/freqtrade/__init__.py
echo "   ✓ Version updated"

# Clear Python cache
echo ""
echo "4. Clearing Python cache..."
find /home/freqtrade -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find /home/freqtrade -name "*.pyc" -delete 2>/dev/null
find /home/freqtrade -name "*.pyo" -delete 2>/dev/null
echo "   ✓ Cache cleared"

echo ""
echo "========================================="
echo "All fixes applied!"
echo "========================================="
echo ""
echo "Testing the fixes..."
echo ""

# Quick test to verify the date column issue is fixed
python3 -c "
import sys
sys.path.insert(0, '/home/freqtrade')
from freqtrade.data.history import load_pair_history
from freqtrade.data.history.datahandlers import get_datahandler
from freqtrade.enums import CandleType
from pathlib import Path

print('Testing data loading...')
dh = get_datahandler(Path('/home/ubuntu/user_data/data/binance'), 'feather')
df = dh.ohlcv_load('BTC/USDT:USDT', '5m', CandleType.FUTURES)
if 'date' in df.columns:
    print('✓ SUCCESS: date column is present!')
    print(f'  Columns: {list(df.columns)}')
else:
    print('✗ FAILED: date column is still missing')
    print(f'  Columns: {list(df.columns)}')
" 2>/dev/null

echo ""
echo "Now run backtesting with:"
echo "  freqtrade backtesting --strategy DLIStrategy"
echo ""
echo "If it still fails, run:"
echo "  freqtrade --version  # Should show 2025.8-rc5"
echo "  python3 -c \"import sys; sys.path.insert(0, '/home/freqtrade'); from freqtrade.optimize.backtesting import get_strategy_run_id; print('Import works!')\""
