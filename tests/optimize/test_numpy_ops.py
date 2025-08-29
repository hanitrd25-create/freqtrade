"""Test NumPy optimized operations."""
import numpy as np
import pandas as pd
import pytest

from freqtrade.optimize import numpy_ops


def test_shift_series():
    """Test shift operation on pandas Series."""
    data = pd.Series([1, 2, 3, 4, 5])
    
    # Test forward shift
    result = numpy_ops.shift(data, 1)
    expected = pd.Series([np.nan, 1, 2, 3, 4])
    pd.testing.assert_series_equal(result, expected)
    
    # Test backward shift
    result = numpy_ops.shift(data, -1)
    expected = pd.Series([2, 3, 4, 5, np.nan])
    pd.testing.assert_series_equal(result, expected)
    
    # Test with fill value
    result = numpy_ops.shift(data, 2, fill_value=0)
    expected = pd.Series([0, 0, 1, 2, 3])
    pd.testing.assert_series_equal(result, expected)


def test_shift_dataframe():
    """Test shift operation on pandas DataFrame."""
    data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    
    result = numpy_ops.shift(data, 1)
    expected = pd.DataFrame({'A': [np.nan, 1, 2], 'B': [np.nan, 4, 5]})
    pd.testing.assert_frame_equal(result, expected)


def test_shift_numpy_array():
    """Test shift operation on numpy array."""
    data = np.array([1, 2, 3, 4, 5])
    
    result = numpy_ops.shift(data, 1)
    expected = np.array([np.nan, 1, 2, 3, 4])
    np.testing.assert_array_equal(result, expected)


def test_rolling_mean():
    """Test rolling mean calculation."""
    data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    result = numpy_ops.rolling_mean(data, window=3)
    # Calculate expected using pandas for comparison
    expected = data.rolling(window=3).mean().values
    np.testing.assert_array_almost_equal(result, expected)
    
    # Test with min_periods
    result = numpy_ops.rolling_mean(data, window=3, min_periods=2)
    expected = data.rolling(window=3, min_periods=2).mean().values
    np.testing.assert_array_almost_equal(result, expected)


def test_rolling_std():
    """Test rolling standard deviation calculation."""
    data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    result = numpy_ops.rolling_std(data, window=3)
    expected = data.rolling(window=3).std().values
    np.testing.assert_array_almost_equal(result, expected)


def test_rolling_max():
    """Test rolling maximum calculation."""
    data = pd.Series([3, 1, 4, 1, 5, 9, 2, 6, 5, 3])
    
    result = numpy_ops.rolling_max(data, window=3)
    expected = data.rolling(window=3).max().values
    np.testing.assert_array_almost_equal(result, expected)


def test_rolling_min():
    """Test rolling minimum calculation."""
    data = pd.Series([3, 1, 4, 1, 5, 9, 2, 6, 5, 3])
    
    result = numpy_ops.rolling_min(data, window=3)
    expected = data.rolling(window=3).min().values
    np.testing.assert_array_almost_equal(result, expected)


def test_diff():
    """Test diff operation."""
    data = pd.Series([1, 3, 6, 10, 15])
    
    result = numpy_ops.diff(data, periods=1)
    expected = data.diff(periods=1)
    pd.testing.assert_series_equal(result, expected)
    
    # Test with periods > 1
    result = numpy_ops.diff(data, periods=2)
    expected = data.diff(periods=2)
    pd.testing.assert_series_equal(result, expected)


def test_pct_change():
    """Test percentage change calculation."""
    data = pd.Series([100, 110, 120, 115, 130])
    
    result = numpy_ops.pct_change(data, periods=1)
    expected = data.pct_change(periods=1)
    pd.testing.assert_series_equal(result, expected)
    
    # Test with zero values
    data_with_zero = pd.Series([100, 0, 120, 115, 130])
    result = numpy_ops.pct_change(data_with_zero, periods=1)
    expected = data_with_zero.pct_change(periods=1)
    pd.testing.assert_series_equal(result, expected)


def test_performance_comparison():
    """Test that NumPy operations are faster than pandas for large datasets."""
    import time
    
    # Create large dataset
    large_data = pd.Series(np.random.randn(10000))
    
    # Time pandas shift
    start = time.perf_counter()
    for _ in range(100):
        _ = large_data.shift(1)
    pandas_time = time.perf_counter() - start
    
    # Time numpy shift
    start = time.perf_counter()
    for _ in range(100):
        _ = numpy_ops.shift(large_data, 1)
    numpy_time = time.perf_counter() - start
    
    # NumPy should be at least as fast, often faster
    # We allow some tolerance since performance can vary
    assert numpy_time < pandas_time * 1.5, f"NumPy shift slower than expected: {numpy_time:.4f}s vs {pandas_time:.4f}s"
