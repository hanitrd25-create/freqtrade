"""
Tests for the indicator cache module.
"""

import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from freqtrade.optimize.indicator_cache import (
    IndicatorCache, SignalCache, cached_indicator,
    get_indicator_cache, get_signal_cache, clear_all_caches
)


@pytest.fixture
def sample_dataframe():
    """Create a sample dataframe for testing."""
    dates = pd.date_range('2024-01-01', periods=100, freq='1h')
    return pd.DataFrame({
        'date': dates,
        'open': np.random.randn(100) + 100,
        'high': np.random.randn(100) + 101,
        'low': np.random.randn(100) + 99,
        'close': np.random.randn(100) + 100,
        'volume': np.random.randn(100) * 1000 + 10000,
    })


@pytest.fixture
def indicator_cache():
    """Create a fresh indicator cache."""
    return IndicatorCache(max_cache_size=10)


@pytest.fixture
def signal_cache():
    """Create a fresh signal cache."""
    return SignalCache(max_cache_size=10)


class TestIndicatorCache:
    """Test the IndicatorCache class."""
    
    def test_cache_initialization(self):
        """Test cache initialization."""
        cache = IndicatorCache(max_cache_size=50)
        assert cache.max_cache_size == 50
        assert len(cache.cache) == 0
        assert cache.cache_hits == 0
        assert cache.cache_misses == 0
    
    def test_generate_cache_key(self, indicator_cache, sample_dataframe):
        """Test cache key generation."""
        params = {'period': 14, 'method': 'sma'}
        key1 = indicator_cache._generate_cache_key(
            sample_dataframe, 'rsi', params
        )
        
        # Same inputs should generate same key
        key2 = indicator_cache._generate_cache_key(
            sample_dataframe, 'rsi', params
        )
        assert key1 == key2
        
        # Different indicator name should generate different key
        key3 = indicator_cache._generate_cache_key(
            sample_dataframe, 'macd', params
        )
        assert key1 != key3
        
        # Different params should generate different key
        params2 = {'period': 21, 'method': 'sma'}
        key4 = indicator_cache._generate_cache_key(
            sample_dataframe, 'rsi', params2
        )
        assert key1 != key4
        
        # Different dataframe should generate different key
        df2 = sample_dataframe.copy()
        df2['close'] = df2['close'] * 2
        key5 = indicator_cache._generate_cache_key(
            df2, 'rsi', params
        )
        assert key1 != key5
    
    def test_cache_hit_and_miss(self, indicator_cache, sample_dataframe):
        """Test cache hits and misses."""
        def calculate_sma(df, period=14):
            result = df.copy()
            result['sma'] = df['close'].rolling(period).mean()
            return result
        
        params = {'period': 14}
        
        # First call should be a miss
        result1 = indicator_cache.get(
            sample_dataframe, 'sma', params, calculate_sma
        )
        assert indicator_cache.cache_misses == 1
        assert indicator_cache.cache_hits == 0
        assert 'sma' in result1.columns
        
        # Second call with same params should be a hit
        result2 = indicator_cache.get(
            sample_dataframe, 'sma', params, calculate_sma
        )
        assert indicator_cache.cache_misses == 1
        assert indicator_cache.cache_hits == 1
        assert result1.equals(result2)
        
        # Call with different params should be a miss
        params2 = {'period': 21}
        result3 = indicator_cache.get(
            sample_dataframe, 'sma', params2, calculate_sma
        )
        assert indicator_cache.cache_misses == 2
        assert indicator_cache.cache_hits == 1
    
    def test_cache_eviction(self, sample_dataframe):
        """Test cache eviction when max size is reached."""
        cache = IndicatorCache(max_cache_size=3)
        
        def dummy_calc(df, **kwargs):
            return df.copy()
        
        # Fill cache to max size
        for i in range(3):
            params = {'param': i}
            cache.get(sample_dataframe, f'ind_{i}', params, dummy_calc)
        
        assert len(cache.cache) == 3
        
        # Adding one more should evict the oldest
        params = {'param': 3}
        cache.get(sample_dataframe, 'ind_3', params, dummy_calc)
        
        assert len(cache.cache) == 3
        # First entry should have been evicted
        key_0 = cache._generate_cache_key(sample_dataframe, 'ind_0', {'param': 0})
        assert key_0 not in cache.cache
    
    def test_cache_clear(self, indicator_cache, sample_dataframe):
        """Test clearing the cache."""
        def dummy_calc(df, **kwargs):
            return df.copy()
        
        # Add some entries
        for i in range(5):
            params = {'param': i}
            indicator_cache.get(sample_dataframe, f'ind_{i}', params, dummy_calc)
        
        assert len(indicator_cache.cache) > 0
        assert indicator_cache.cache_misses > 0
        
        # Clear cache
        indicator_cache.clear()
        
        assert len(indicator_cache.cache) == 0
        assert indicator_cache.cache_hits == 0
        assert indicator_cache.cache_misses == 0
    
    def test_cache_stats(self, indicator_cache, sample_dataframe):
        """Test cache statistics."""
        def dummy_calc(df, **kwargs):
            return df.copy()
        
        # Generate some hits and misses
        params = {'param': 1}
        indicator_cache.get(sample_dataframe, 'ind', params, dummy_calc)  # miss
        indicator_cache.get(sample_dataframe, 'ind', params, dummy_calc)  # hit
        indicator_cache.get(sample_dataframe, 'ind', params, dummy_calc)  # hit
        
        stats = indicator_cache.get_stats()
        
        assert stats['cache_size'] == 1
        assert stats['max_size'] == 10
        assert stats['hits'] == 2
        assert stats['misses'] == 1
        assert stats['total_requests'] == 3
        assert stats['hit_rate'] == 2/3


class TestCachedIndicatorDecorator:
    """Test the cached_indicator decorator."""
    
    def test_decorator_basic(self, sample_dataframe):
        """Test basic decorator functionality."""
        cache = IndicatorCache()
        
        @cached_indicator(cache)
        def calculate_sma(dataframe, period=14):
            result = dataframe.copy()
            result['sma'] = dataframe['close'].rolling(period).mean()
            return result
        
        # First call
        result1 = calculate_sma(sample_dataframe, period=14)
        assert cache.cache_misses == 1
        assert cache.cache_hits == 0
        
        # Second call should hit cache
        result2 = calculate_sma(sample_dataframe, period=14)
        assert cache.cache_misses == 1
        assert cache.cache_hits == 1
        assert result1.equals(result2)
    
    def test_decorator_without_cache(self, sample_dataframe):
        """Test decorator creates its own cache if none provided."""
        @cached_indicator()
        def calculate_ema(dataframe, period=14):
            result = dataframe.copy()
            result['ema'] = dataframe['close'].ewm(span=period).mean()
            return result
        
        # Should work without explicit cache
        result = calculate_ema(sample_dataframe, period=14)
        assert 'ema' in result.columns
        
        # Cache should be accessible via function attribute
        assert hasattr(calculate_ema, 'cache')
        assert isinstance(calculate_ema.cache, IndicatorCache)


class TestSignalCache:
    """Test the SignalCache class."""
    
    def test_signal_cache_initialization(self):
        """Test signal cache initialization."""
        cache = SignalCache(max_cache_size=25)
        assert cache.max_cache_size == 25
        assert len(cache.cache) == 0
        assert cache.cache_hits == 0
        assert cache.cache_misses == 0
    
    def test_signal_cache_key_generation(self, signal_cache, sample_dataframe):
        """Test signal cache key generation."""
        # Add some indicators to dataframe
        df = sample_dataframe.copy()
        df['rsi'] = np.random.randn(100)
        df['macd'] = np.random.randn(100)
        
        params = {'buy_rsi': 30, 'sell_rsi': 70}
        key1 = signal_cache._generate_cache_key(df, 'TestStrategy', params)
        
        # Same inputs should generate same key
        key2 = signal_cache._generate_cache_key(df, 'TestStrategy', params)
        assert key1 == key2
        
        # Different strategy should generate different key
        key3 = signal_cache._generate_cache_key(df, 'OtherStrategy', params)
        assert key1 != key3
        
        # Different params should generate different key
        params2 = {'buy_rsi': 25, 'sell_rsi': 75}
        key4 = signal_cache._generate_cache_key(df, 'TestStrategy', params2)
        assert key1 != key4
    
    def test_signal_cache_hit_and_miss(self, signal_cache, sample_dataframe):
        """Test signal cache hits and misses."""
        df = sample_dataframe.copy()
        df['rsi'] = np.random.randn(100)
        
        def calculate_signals(dataframe, buy_rsi=30, sell_rsi=70):
            buy_signals = dataframe['rsi'] < buy_rsi
            sell_signals = dataframe['rsi'] > sell_rsi
            return buy_signals, sell_signals
        
        params = {'buy_rsi': 30, 'sell_rsi': 70}
        
        # First call should be a miss
        buy1, sell1 = signal_cache.get(df, 'TestStrategy', params, calculate_signals)
        assert signal_cache.cache_misses == 1
        assert signal_cache.cache_hits == 0
        
        # Second call should be a hit
        buy2, sell2 = signal_cache.get(df, 'TestStrategy', params, calculate_signals)
        assert signal_cache.cache_misses == 1
        assert signal_cache.cache_hits == 1
        assert buy1.equals(buy2)
        assert sell1.equals(sell2)
    
    def test_signal_cache_clear(self, signal_cache, sample_dataframe):
        """Test clearing signal cache."""
        df = sample_dataframe.copy()
        df['rsi'] = np.random.randn(100)
        
        def dummy_signals(dataframe, **kwargs):
            return dataframe['rsi'] > 0, dataframe['rsi'] < 0
        
        # Add some entries
        for i in range(3):
            params = {'param': i}
            signal_cache.get(df, f'Strategy_{i}', params, dummy_signals)
        
        assert len(signal_cache.cache) > 0
        
        # Clear cache
        signal_cache.clear()
        
        assert len(signal_cache.cache) == 0
        assert signal_cache.cache_hits == 0
        assert signal_cache.cache_misses == 0


class TestGlobalCaches:
    """Test global cache functions."""
    
    def test_get_indicator_cache(self):
        """Test getting global indicator cache."""
        cache1 = get_indicator_cache()
        cache2 = get_indicator_cache()
        
        # Should return same instance
        assert cache1 is cache2
        assert isinstance(cache1, IndicatorCache)
    
    def test_get_signal_cache(self):
        """Test getting global signal cache."""
        cache1 = get_signal_cache()
        cache2 = get_signal_cache()
        
        # Should return same instance
        assert cache1 is cache2
        assert isinstance(cache1, SignalCache)
    
    @patch('freqtrade.optimize.indicator_cache._global_indicator_cache')
    @patch('freqtrade.optimize.indicator_cache._global_signal_cache')
    def test_clear_all_caches(self, mock_signal_cache, mock_indicator_cache):
        """Test clearing all global caches."""
        # Setup mocks
        mock_ind = MagicMock()
        mock_sig = MagicMock()
        mock_indicator_cache.__bool__.return_value = True
        mock_signal_cache.__bool__.return_value = True
        mock_indicator_cache.clear = mock_ind
        mock_signal_cache.clear = mock_sig
        
        # Clear all caches
        clear_all_caches()
        
        # Both caches should be cleared
        mock_ind.assert_called_once()
        mock_sig.assert_called_once()
