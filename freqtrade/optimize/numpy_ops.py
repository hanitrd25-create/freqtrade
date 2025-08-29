"""
NumPy-optimized DataFrame operations for performance improvement.
Provides drop-in replacements for common pandas operations using NumPy.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional


def shift(data: Union[pd.Series, pd.DataFrame, np.ndarray], 
          periods: int = 1, 
          fill_value: Optional[float] = None) -> Union[pd.Series, pd.DataFrame, np.ndarray]:
    """
    NumPy-optimized shift operation.
    
    :param data: Input data (Series, DataFrame, or ndarray)
    :param periods: Number of periods to shift (positive = forward, negative = backward)
    :param fill_value: Value to use for filling NaN values (default: np.nan)
    :return: Shifted data in the same format as input
    """
    if fill_value is None:
        fill_value = np.nan
    
    if isinstance(data, pd.DataFrame):
        # Handle DataFrame column-wise
        result_data = np.empty_like(data.values, dtype=np.float64)
        for i in range(data.shape[1]):
            result_data[:, i] = _shift_array(data.iloc[:, i].values, periods, fill_value)
        return pd.DataFrame(result_data, index=data.index, columns=data.columns)
    
    elif isinstance(data, pd.Series):
        # Handle Series
        shifted = _shift_array(data.values, periods, fill_value)
        return pd.Series(shifted, index=data.index, name=data.name)
    
    else:
        # Handle numpy array
        return _shift_array(data, periods, fill_value)


def _shift_array(arr: np.ndarray, periods: int, fill_value: float) -> np.ndarray:
    """Internal function to shift a numpy array."""
    # Ensure result array can hold NaN values
    if arr.dtype.kind in 'iu' and np.isnan(fill_value):  # integer types
        result = np.empty(arr.shape, dtype=np.float64)
        result[:] = arr
        arr = result
        result = np.empty_like(arr)
    else:
        result = np.empty_like(arr)
    
    if periods > 0:
        # Shift forward (down)
        result[:periods] = fill_value
        result[periods:] = arr[:-periods]
    elif periods < 0:
        # Shift backward (up)
        result[periods:] = fill_value
        result[:periods] = arr[-periods:]
    else:
        # No shift
        result[:] = arr
    
    return result


def rolling_mean(data: Union[pd.Series, np.ndarray], 
                 window: int,
                 min_periods: Optional[int] = None) -> np.ndarray:
    """
    NumPy-optimized rolling mean calculation.
    
    :param data: Input data (Series or ndarray)
    :param window: Size of the rolling window
    :param min_periods: Minimum number of observations in window (default: window)
    :return: Array with rolling mean values
    """
    if isinstance(data, pd.Series):
        arr = data.values
    else:
        arr = data
    
    if min_periods is None:
        min_periods = window
    
    return _rolling_window_mean(arr, window, min_periods)


def rolling_std(data: Union[pd.Series, np.ndarray], 
                window: int,
                min_periods: Optional[int] = None,
                ddof: int = 1) -> np.ndarray:
    """
    NumPy-optimized rolling standard deviation calculation.
    
    :param data: Input data (Series or ndarray)
    :param window: Size of the rolling window
    :param min_periods: Minimum number of observations in window (default: window)
    :param ddof: Delta degrees of freedom (default: 1)
    :return: Array with rolling std values
    """
    if isinstance(data, pd.Series):
        arr = data.values
    else:
        arr = data
    
    if min_periods is None:
        min_periods = window
    
    return _rolling_window_std(arr, window, min_periods, ddof)


def rolling_max(data: Union[pd.Series, np.ndarray], 
                window: int,
                min_periods: Optional[int] = None) -> np.ndarray:
    """
    NumPy-optimized rolling maximum calculation.
    
    :param data: Input data (Series or ndarray)
    :param window: Size of the rolling window
    :param min_periods: Minimum number of observations in window (default: window)
    :return: Array with rolling max values
    """
    if isinstance(data, pd.Series):
        arr = data.values
    else:
        arr = data
    
    if min_periods is None:
        min_periods = window
    
    return _rolling_window_max(arr, window, min_periods)


def rolling_min(data: Union[pd.Series, np.ndarray], 
                window: int,
                min_periods: Optional[int] = None) -> np.ndarray:
    """
    NumPy-optimized rolling minimum calculation.
    
    :param data: Input data (Series or ndarray)
    :param window: Size of the rolling window
    :param min_periods: Minimum number of observations in window (default: window)
    :return: Array with rolling min values
    """
    if isinstance(data, pd.Series):
        arr = data.values
    else:
        arr = data
    
    if min_periods is None:
        min_periods = window
    
    return _rolling_window_min(arr, window, min_periods)


def _rolling_window_mean(arr: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    """Calculate rolling mean using NumPy stride tricks for efficiency."""
    n = len(arr)
    result = np.full(n, np.nan)
    
    if n < min_periods:
        return result
    
    # Use cumsum for O(n) complexity instead of O(n*window)
    cumsum = np.concatenate(([0], np.nancumsum(arr)))
    
    # Calculate rolling sum efficiently
    for i in range(min_periods - 1, n):
        start_idx = max(0, i - window + 1)
        window_sum = cumsum[i + 1] - cumsum[start_idx]
        
        # Count non-nan values in window
        window_data = arr[start_idx:i + 1]
        valid_count = np.sum(~np.isnan(window_data))
        
        if valid_count >= min_periods:
            result[i] = window_sum / valid_count
    
    return result


def _rolling_window_std(arr: np.ndarray, window: int, min_periods: int, ddof: int) -> np.ndarray:
    """Calculate rolling standard deviation using NumPy."""
    n = len(arr)
    result = np.full(n, np.nan)
    
    if n < min_periods:
        return result
    
    for i in range(min_periods - 1, n):
        start_idx = max(0, i - window + 1)
        window_data = arr[start_idx:i + 1]
        
        # Remove NaN values
        valid_data = window_data[~np.isnan(window_data)]
        
        if len(valid_data) >= min_periods:
            result[i] = np.std(valid_data, ddof=ddof)
    
    return result


def _rolling_window_max(arr: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    """Calculate rolling maximum using NumPy."""
    n = len(arr)
    result = np.full(n, np.nan)
    
    if n < min_periods:
        return result
    
    for i in range(min_periods - 1, n):
        start_idx = max(0, i - window + 1)
        window_data = arr[start_idx:i + 1]
        
        # Remove NaN values
        valid_data = window_data[~np.isnan(window_data)]
        
        if len(valid_data) >= min_periods:
            result[i] = np.max(valid_data)
    
    return result


def _rolling_window_min(arr: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    """Calculate rolling minimum using NumPy."""
    n = len(arr)
    result = np.full(n, np.nan)
    
    if n < min_periods:
        return result
    
    for i in range(min_periods - 1, n):
        start_idx = max(0, i - window + 1)
        window_data = arr[start_idx:i + 1]
        
        # Remove NaN values
        valid_data = window_data[~np.isnan(window_data)]
        
        if len(valid_data) >= min_periods:
            result[i] = np.min(valid_data)
    
    return result


def diff(data: Union[pd.Series, pd.DataFrame, np.ndarray], 
         periods: int = 1) -> Union[pd.Series, pd.DataFrame, np.ndarray]:
    """
    NumPy-optimized diff operation (current - previous).
    
    :param data: Input data (Series, DataFrame, or ndarray)
    :param periods: Periods to shift for calculating difference
    :return: Difference array in the same format as input
    """
    if isinstance(data, pd.DataFrame):
        # Handle DataFrame column-wise
        result_data = np.empty_like(data.values)
        for i in range(data.shape[1]):
            result_data[:, i] = _diff_array(data.iloc[:, i].values, periods)
        return pd.DataFrame(result_data, index=data.index, columns=data.columns)
    
    elif isinstance(data, pd.Series):
        # Handle Series
        diff_values = _diff_array(data.values, periods)
        return pd.Series(diff_values, index=data.index, name=data.name)
    
    else:
        # Handle numpy array
        return _diff_array(data, periods)


def _diff_array(arr: np.ndarray, periods: int) -> np.ndarray:
    """Internal function to calculate difference of a numpy array."""
    # Ensure result array can hold NaN values
    if arr.dtype.kind in 'iu':  # integer types
        result = np.empty(arr.shape, dtype=np.float64)
    else:
        result = np.empty_like(arr)
    
    result[:periods] = np.nan
    result[periods:] = arr[periods:] - arr[:-periods]
    return result


def pct_change(data: Union[pd.Series, pd.DataFrame, np.ndarray], 
               periods: int = 1) -> Union[pd.Series, pd.DataFrame, np.ndarray]:
    """
    NumPy-optimized percentage change calculation.
    
    :param data: Input data (Series, DataFrame, or ndarray)
    :param periods: Periods to shift for calculating percentage change
    :return: Percentage change in the same format as input
    """
    if isinstance(data, pd.DataFrame):
        # Handle DataFrame column-wise
        result_data = np.empty_like(data.values)
        for i in range(data.shape[1]):
            result_data[:, i] = _pct_change_array(data.iloc[:, i].values, periods)
        return pd.DataFrame(result_data, index=data.index, columns=data.columns)
    
    elif isinstance(data, pd.Series):
        # Handle Series
        pct_values = _pct_change_array(data.values, periods)
        return pd.Series(pct_values, index=data.index, name=data.name)
    
    else:
        # Handle numpy array
        return _pct_change_array(data, periods)


def _pct_change_array(arr: np.ndarray, periods: int) -> np.ndarray:
    """Internal function to calculate percentage change of a numpy array."""
    # Ensure result array can hold NaN values
    if arr.dtype.kind in 'iu':  # integer types
        result = np.empty(arr.shape, dtype=np.float64)
        arr = arr.astype(np.float64)  # Convert for division
    else:
        result = np.empty_like(arr)
    
    result[:periods] = np.nan
    
    # Calculate percentage change (pandas compatible behavior)
    prev_values = arr[:-periods]
    with np.errstate(divide='ignore', invalid='ignore'):
        result[periods:] = (arr[periods:] - prev_values) / prev_values
    
    return result
