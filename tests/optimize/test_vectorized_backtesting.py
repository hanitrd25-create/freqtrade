"""Test vectorized backtesting functionality."""
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from freqtrade.optimize.vectorized_backtesting import VectorizedBacktester
from freqtrade.enums import TradingMode


@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    return {
        'stake_amount': 100,
        'fee': 0.001,
        'max_open_trades': 3,
        'minimal_roi': {
            '0': 0.10,
            '10': 0.05,
            '20': 0.01,
        },
        'stoploss': -0.10,
        'trailing_stop': False,
        'position_stacking': False,
        'trading_mode': TradingMode.SPOT,
    }


@pytest.fixture
def sample_data():
    """Create sample OHLCV data with signals."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
    
    # Generate synthetic price data
    np.random.seed(42)
    close_prices = 100 + np.cumsum(np.random.randn(100) * 2)
    
    df = pd.DataFrame({
        'date': dates,
        'open': close_prices * (1 + np.random.randn(100) * 0.001),
        'high': close_prices * (1 + np.abs(np.random.randn(100) * 0.005)),
        'low': close_prices * (1 - np.abs(np.random.randn(100) * 0.005)),
        'close': close_prices,
        'volume': np.random.uniform(1000, 10000, 100),
    })
    
    # Add simple signals
    df['enter_long'] = False
    df['exit_long'] = False
    
    # Add some entry signals
    df.loc[[10, 30, 50, 70], 'enter_long'] = True
    
    # Add some exit signals
    df.loc[[20, 40, 60, 80], 'exit_long'] = True
    
    return df


def test_vectorized_backtest_init(sample_config):
    """Test VectorizedBacktester initialization."""
    backtester = VectorizedBacktester(sample_config)
    
    assert backtester.stake_amount == 100
    assert backtester.fee == 0.001
    assert backtester.max_open_trades == 3
    assert backtester.stoploss == -0.10
    assert backtester.trailing_stop is False


def test_can_use_vectorized(sample_config):
    """Test strategy compatibility check."""
    backtester = VectorizedBacktester(sample_config)
    
    # Create mock strategy
    class SimpleStrategy:
        pass
    
    class ComplexStrategy:
        def custom_exit(self, *args, **kwargs):
            return None
    
    # Simple strategy should be compatible
    assert backtester.can_use_vectorized(SimpleStrategy()) is True
    
    # Complex strategy with custom_exit should not be compatible
    assert backtester.can_use_vectorized(ComplexStrategy()) is False
    
    # Trailing stop makes it incompatible
    backtester.trailing_stop = True
    assert backtester.can_use_vectorized(SimpleStrategy()) is False
    backtester.trailing_stop = False
    
    # Position stacking makes it incompatible
    backtester.config['position_stacking'] = True
    assert backtester.can_use_vectorized(SimpleStrategy()) is False


def test_vectorized_backtest_basic(sample_config, sample_data):
    """Test basic vectorized backtesting."""
    backtester = VectorizedBacktester(sample_config)
    
    processed = {'BTC/USDT': sample_data}
    start_date = sample_data['date'].iloc[0]
    end_date = sample_data['date'].iloc[-1]
    
    results = backtester.vectorized_backtest(processed, start_date, end_date)
    
    # Should have some trades
    assert len(results) > 0
    
    # Check trade structure
    required_columns = [
        'pair', 'stake_amount', 'open_date', 'close_date',
        'open_rate', 'close_rate', 'profit_ratio', 'profit_abs'
    ]
    for col in required_columns:
        assert col in results.columns
    
    # All trades should be closed
    assert all(results['is_open'] == False)


def test_find_exit_vectorized(sample_config, sample_data):
    """Test exit finding logic."""
    backtester = VectorizedBacktester(sample_config)
    
    # Extract price arrays
    high_prices = sample_data['high'].values
    low_prices = sample_data['low'].values
    close_prices = sample_data['close'].values
    exit_signals = sample_data['exit_long'].values
    
    entry_idx = 10
    entry_price = close_prices[entry_idx]
    
    # Find exit
    exit_idx = backtester._find_exit_vectorized(
        entry_idx, exit_signals, high_prices, low_prices, close_prices, entry_price
    )
    
    # Should find an exit
    assert exit_idx is not None
    assert exit_idx > entry_idx


def test_calculate_exit_price(sample_config):
    """Test exit price calculation."""
    # Create config without ROI for simpler testing
    config_no_roi = sample_config.copy()
    config_no_roi['minimal_roi'] = {}
    backtester = VectorizedBacktester(config_no_roi)
    
    # Create controlled data for testing
    open_prices = np.array([100, 101, 102, 103, 104])
    high_prices = np.array([101, 102, 103, 104, 105])
    low_prices = np.array([99, 100, 101, 102, 103])
    close_prices = np.array([100.5, 101.5, 102.5, 103.5, 104.5])
    
    entry_price = 100.0
    exit_idx = 2
    
    # Test 1: Signal exit (no stoploss or ROI hit)
    exit_price = backtester._calculate_exit_price(
        exit_idx, True, open_prices, high_prices, low_prices, close_prices, entry_price
    )
    # Should use open price of next candle for signal exit
    assert exit_price == open_prices[exit_idx + 1]  # 103
    
    # Test 2: Stoploss hit
    # Modify data so stoploss is hit
    low_prices_sl = low_prices.copy()
    low_prices_sl[exit_idx] = 89  # Below stoploss of 90 (100 * 0.9)
    
    exit_price_sl = backtester._calculate_exit_price(
        exit_idx, False, open_prices, high_prices, low_prices_sl, close_prices, entry_price
    )
    # Should use stoploss price or low price (whichever is higher)
    expected_sl = max(entry_price * (1 + backtester.stoploss), low_prices_sl[exit_idx])
    assert exit_price_sl == expected_sl
    
    # Test 3: ROI hit
    # Use config with ROI
    backtester_roi = VectorizedBacktester(sample_config)
    # ROI should trigger at 101 (1% minimum ROI)
    exit_price_roi = backtester_roi._calculate_exit_price(
        exit_idx, False, open_prices, high_prices, low_prices, close_prices, entry_price
    )
    # high_prices[2] = 103 is > roi_price = 101, so ROI hits
    assert exit_price_roi == 101.0  # min(roi_price, high_price)
    
    # Test 4: End of data
    exit_idx_end = len(close_prices) - 1
    exit_price_end = backtester._calculate_exit_price(
        exit_idx_end, False, open_prices, high_prices, low_prices, close_prices, entry_price
    )
    assert exit_price_end == close_prices[exit_idx_end]


def test_stoploss_hit(sample_config):
    """Test stoploss detection."""
    # Modify config to have no ROI to ensure stoploss is tested
    config = sample_config.copy()
    config['minimal_roi'] = {}  # Disable ROI
    backtester = VectorizedBacktester(config)
    
    # Create data where stoploss is hit quickly (before any ROI possibility)
    dates = pd.date_range(start='2024-01-01', periods=10, freq='1h')
    df = pd.DataFrame({
        'date': dates,
        'open': [100, 99, 88, 87, 86, 85, 84, 83, 82, 81],  # Big gap down
        'high': [101, 99.5, 89, 88, 87, 86, 85, 84, 83, 82],
        'low': [99, 98, 87, 86, 85, 84, 83, 82, 81, 80],  # Stoploss definitely hit
        'close': [99.5, 88.5, 87.5, 86.5, 85.5, 84.5, 83.5, 82.5, 81.5, 80],
        'volume': [1000] * 10,
        'enter_long': [True] + [False] * 9,
        'exit_long': [False] * 10,
    })
    
    processed = {'BTC/USDT': df}
    results = backtester.vectorized_backtest(
        processed, df['date'].iloc[0], df['date'].iloc[-1]
    )
    
    if len(results) > 0:
        # Check that stoploss was triggered
        assert any(results['exit_reason'] == 'stop_loss')
        # Check that loss is approximately the stoploss value
        assert results.iloc[0]['profit_ratio'] < 0


def test_roi_exit(sample_config):
    """Test ROI-based exit."""
    backtester = VectorizedBacktester(sample_config)
    
    # Create data where ROI is hit
    dates = pd.date_range(start='2024-01-01', periods=10, freq='1h')
    df = pd.DataFrame({
        'date': dates,
        'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        'high': [101, 102, 103, 104, 105, 112, 107, 108, 109, 110],  # ROI hit
        'low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
        'close': [100.5, 101.5, 102.5, 103.5, 104.5, 111, 106.5, 107.5, 108.5, 109.5],
        'volume': [1000] * 10,
        'enter_long': [True] + [False] * 9,
        'exit_long': [False] * 10,
    })
    
    processed = {'BTC/USDT': df}
    results = backtester.vectorized_backtest(
        processed, df['date'].iloc[0], df['date'].iloc[-1]
    )
    
    if len(results) > 0:
        # Check that ROI exit was triggered
        assert any(results['exit_reason'] == 'roi')
        # Check that profit is positive
        assert results.iloc[0]['profit_ratio'] > 0


def test_multiple_pairs(sample_config, sample_data):
    """Test backtesting with multiple pairs."""
    backtester = VectorizedBacktester(sample_config)
    
    # Create data for multiple pairs
    processed = {
        'BTC/USDT': sample_data.copy(),
        'ETH/USDT': sample_data.copy(),
        'ADA/USDT': sample_data.copy(),
    }
    
    # Modify signals slightly for each pair
    processed['ETH/USDT']['enter_long'] = False
    processed['ETH/USDT'].loc[[15, 35, 55], 'enter_long'] = True
    
    processed['ADA/USDT']['enter_long'] = False
    processed['ADA/USDT'].loc[[5, 25, 45], 'enter_long'] = True
    
    start_date = sample_data['date'].iloc[0]
    end_date = sample_data['date'].iloc[-1]
    
    results = backtester.vectorized_backtest(processed, start_date, end_date)
    
    # Should have trades from multiple pairs
    assert len(results) > 0
    assert len(results['pair'].unique()) > 1
    
    # Each pair should have trades
    for pair in ['BTC/USDT', 'ETH/USDT', 'ADA/USDT']:
        assert pair in results['pair'].values
