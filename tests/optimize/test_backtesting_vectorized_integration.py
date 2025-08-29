"""
Integration tests for vectorized backtesting in the main backtesting flow
"""
import pytest
from datetime import datetime
import pandas as pd
from unittest.mock import MagicMock, patch

from freqtrade.optimize.backtesting import Backtesting
from freqtrade.strategy import IStrategy
from freqtrade.enums import TradingMode


@pytest.fixture
def simple_strategy():
    """Create a simple strategy for testing."""
    strategy = MagicMock(spec=IStrategy)
    strategy.minimal_roi = {"0": 0.1}
    strategy.stoploss = -0.05
    strategy.trailing_stop = False
    strategy.can_short = False
    strategy.use_custom_stoploss = False
    strategy.use_exit_signal = False
    strategy.position_adjustment_enable = False
    strategy.max_entry_position_adjustment = 0
    strategy.config = {
        'stake_currency': 'USDT',
        'dry_run': True,
    }
    return strategy


@pytest.fixture
def backtesting_instance(default_conf, simple_strategy):
    """Create a Backtesting instance for integration testing."""
    config = default_conf.copy()
    config.update({
        'strategy': 'SampleStrategy',
        'timeframe': '1h',
        'dry_run': True,
        'use_vectorized_backtesting': True,
        'position_stacking': False,
        'trading_mode': TradingMode.SPOT,
        'stake_currency': 'USDT',
        'exchange': {
            'name': 'binance',
            'pair_whitelist': ['BTC/USDT', 'ETH/USDT'],
        }
    })
    
    # Mock exchange
    mock_exchange = MagicMock()
    mock_exchange.name = 'binance'
    mock_exchange.precisionMode = 2
    mock_exchange.precision_mode_price = 2
    mock_exchange.get_fee.return_value = 0.001
    mock_exchange.validate_required_startup_candles.return_value = None
    mock_exchange.markets = {
        'BTC/USDT': {'symbol': 'BTC/USDT', 'base': 'BTC', 'quote': 'USDT', 'active': True},
        'ETH/USDT': {'symbol': 'ETH/USDT', 'base': 'ETH', 'quote': 'USDT', 'active': True},
    }
    
    # Mock PairListManager
    mock_pairlist = MagicMock()
    mock_pairlist.whitelist = ['BTC/USDT', 'ETH/USDT']
    mock_pairlist.name_list = []
    
    with patch('freqtrade.optimize.backtesting.ExchangeResolver.load_exchange', return_value=mock_exchange):
        with patch('freqtrade.optimize.backtesting.StrategyResolver.load_strategy', return_value=simple_strategy):
            with patch('freqtrade.optimize.backtesting.migrate_data'):
                with patch('freqtrade.optimize.backtesting.PairListManager', return_value=mock_pairlist):
                    bt = Backtesting(config)
                    bt.wallets = MagicMock()
                    bt.wallets.get_total.return_value = 1000
                    bt.wallets.update.return_value = None
                    return bt


def test_vectorized_integration_automatic_selection(backtesting_instance, simple_strategy):
    """Test that vectorized backtesting is automatically selected for simple strategies."""
    bt = backtesting_instance
    bt.strategy = simple_strategy
    
    # Ensure config allows vectorized backtesting
    bt.config['use_exit_signal'] = False
    bt.enable_protections = False
    bt.use_vectorized = True
    
    # Create sample data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
    processed = {
        'BTC/USDT': pd.DataFrame({
            'date': dates,
            'open': [100] * 100,
            'high': [101] * 100,
            'low': [99] * 100,
            'close': [100.5] * 100,
            'volume': [1000] * 100,
            'enter_long': [i % 20 == 0 for i in range(100)],
            'exit_long': [False] * 100,
        })
    }
    
    # Mock the vectorized backtester
    with patch.object(bt.vectorized_backtester, 'can_use_vectorized', return_value=True):
        with patch.object(bt.vectorized_backtester, 'vectorized_backtest') as mock_vectorized:
            mock_vectorized.return_value = pd.DataFrame({
                'pair': ['BTC/USDT'],
                'profit_ratio': [0.01],
                'profit_abs': [1.0],
            })
            
            result = bt.backtest(
                processed,
                datetime(2024, 1, 1),
                datetime(2024, 1, 4)
            )
            
            # Verify vectorized backtest was called
            mock_vectorized.assert_called_once()
            assert 'results' in result
            assert len(result['results']) > 0


def test_vectorized_integration_fallback_complex_strategy(backtesting_instance):
    """Test that complex strategies fall back to standard backtesting."""
    bt = backtesting_instance
    
    # Create a complex strategy that should not use vectorized
    complex_strategy = MagicMock(spec=IStrategy)
    complex_strategy.trailing_stop = True  # Makes it complex
    complex_strategy.can_short = False
    complex_strategy.position_adjustment_enable = False
    complex_strategy.config = {'stake_currency': 'USDT'}
    bt.strategy = complex_strategy
    
    # Create sample data
    dates = pd.date_range(start='2024-01-01', periods=10, freq='1h')
    processed = {
        'BTC/USDT': pd.DataFrame({
            'date': dates,
            'open': [100] * 10,
            'high': [101] * 10,
            'low': [99] * 10,
            'close': [100.5] * 10,
            'volume': [1000] * 10,
            'enter_long': [True, False, False, False, False, False, False, False, False, False],
            'exit_long': [False] * 10,
        })
    }
    
    # Mock the vectorized backtester to ensure it's not used
    with patch.object(bt.vectorized_backtester, 'can_use_vectorized', return_value=False):
        with patch.object(bt.vectorized_backtester, 'vectorized_backtest') as mock_vectorized:
            with patch.object(bt, 'prepare_backtest'):
                with patch.object(bt, '_get_ohlcv_as_lists', return_value={}):
                    with patch.object(bt, 'time_pair_generator', return_value=[]):
                        with patch.object(bt, 'handle_left_open'):
                            result = bt.backtest(
                                processed,
                                datetime(2024, 1, 1),
                                datetime(2024, 1, 2)
                            )
                            
                            # Verify vectorized backtest was NOT called
                            mock_vectorized.assert_not_called()


def test_vectorized_integration_config_disable(backtesting_instance, simple_strategy):
    """Test that vectorized backtesting can be disabled via config."""
    bt = backtesting_instance
    bt.use_vectorized = False  # Disable vectorized
    bt.strategy = simple_strategy
    
    # Create sample data
    dates = pd.date_range(start='2024-01-01', periods=10, freq='1h')
    processed = {
        'BTC/USDT': pd.DataFrame({
            'date': dates,
            'open': [100] * 10,
            'high': [101] * 10,
            'low': [99] * 10,
            'close': [100.5] * 10,
            'volume': [1000] * 10,
            'enter_long': [True] + [False] * 9,
            'exit_long': [False] * 10,
        })
    }
    
    # Even though strategy is simple, vectorized should not be used
    with patch.object(bt.vectorized_backtester, 'vectorized_backtest') as mock_vectorized:
        with patch.object(bt, 'prepare_backtest'):
            with patch.object(bt, '_get_ohlcv_as_lists', return_value={}):
                with patch.object(bt, 'time_pair_generator', return_value=[]):
                    with patch.object(bt, 'handle_left_open'):
                        result = bt.backtest(
                            processed,
                            datetime(2024, 1, 1),
                            datetime(2024, 1, 2)
                        )
                        
                        # Verify vectorized backtest was NOT called
                        mock_vectorized.assert_not_called()
